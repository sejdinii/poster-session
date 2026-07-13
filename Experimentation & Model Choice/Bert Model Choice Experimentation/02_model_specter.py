import os
import time
import pandas as pd
import torch
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from pathlib import Path #so the paths work no matter where the script is run from

#duplicate detection experiment for SPECTER2
#
#SPECTER2 differs from all-MiniLM-L6-v2 in two important ways:
#
#1. loading: SPECTER2 does not load as a simple SentenceTransformer
#   it loads as a base model (allenai/specter2_base) plus a separate
#   retrieval adapter (allenai/specter2, the proximity adapter)
#   we embed by tokenising the text, running the model, and taking
#   the first token (CLS) of the last hidden state as the embedding
#
#2. distance metric: SPECTER2 was trained with a triplet margin loss
#   based on EUCLIDEAN distance, not cosine
#   so we rank papers by euclidean distance (smaller distance = more similar)
#   this is confirmed from the SPECTER2 paper and the Ai2 blog
#
#everything else is the same as the minilm experiment:
#reads however many paraphrases are in the csv, ranks against the full dataset
#saves two csvs:
# - results_specter2.csv: per-paper ranks plus the paraphrase and original text
#   so high-rank misses can be inspected to see why the model missed
# - summary_all_models.csv: appends one row, shared with the other 2 models
#
#note: we use abstract only (no title) for all models to keep the comparison fair

#MODEL SETUP
base_model_name = "allenai/specter2_base"
adapter_name = "allenai/specter2"  #the proximity / retrieval adapter
model_label = "SPECTER2"  #label used in the shared summary csv

#PATHS
#anchor everything to this scripts own location instead of wherever the
#terminal happens to be standing. the two inputs live in sibling folders,
#the outputs land in folders created right next to this script.
#if an input lives somewhere else, change only its line here
BASE_DIR = Path(__file__).resolve().parent   #...\Bert Model Choice Experimentation

neurips_path = BASE_DIR.parent / "Neurips Dataset" / "Dataset" / "neurips_accepted_submissions.csv"
paraphrase_path = BASE_DIR.parent / "Neurips Dataset (Summarized)" / "Summarized Dataset" / "paraphrased_abstracts.csv"

#this models own result folder, created automatically if its missing
OUTPUT_DIR = BASE_DIR / "02 Specter2 Result"
OUTPUT_DIR.mkdir(exist_ok=True)
output_path = OUTPUT_DIR / "results_specter2.csv"

#the shared summary gets its own folder too - all 3 model scripts write
#into the same csv there, one row each
SUMMARY_DIR = BASE_DIR / "Summary of Results"
SUMMARY_DIR.mkdir(exist_ok=True)
shared_summary_path = SUMMARY_DIR / "summary_all_models.csv"

#stop early with a clear message if an input file isnt where we expect it
if not neurips_path.exists():
    print(f"could not find the neurips dataset here: {neurips_path}")
    print("run dataset_API.py first, or fix the neurips_path line at the top of this script.")
    raise SystemExit(1)
if not paraphrase_path.exists():
    print(f"could not find the paraphrases here: {paraphrase_path}")
    print("run summarization_API.py first, or fix the paraphrase_path line at the top of this script.")
    raise SystemExit(1)

#how many abstracts we process at once when embedding
#smaller batch because specter2 is a bigger model and your gpu has 6gb
batch_size = 16

#CHECKING GPU
#this experiment must run on the gpu
#if cuda is not available we stop instead of falling back to cpu
#because cpu timings would make the compute comparison meaningless
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Stopping. Fix the GPU setup before running this experiment.")

device = "cuda"
print(f"CUDA available: True")
print(f"GPU: {torch.cuda.get_device_name(0)}")

#LOADING THE MODEL
#first the tokenizer, then the base model, then we attach the retrieval adapter
print("\nLoading SPECTER2...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoAdapterModel.from_pretrained(base_model_name)

#load_as gives the adapter a name, set_active makes it the one being used
model.load_adapter(adapter_name, source="hf", load_as="proximity", set_active=True)

#explicitly activate the adapter for the forward pass
#the set_active=True above does not always stick on its own
#so we set it directly here to make sure the adapter is actually used
model.set_active_adapters("proximity")

#move the model onto the gpu and set it to evaluation mode
#eval mode turns off training-only behaviour like dropout
model.to(device)
model.eval()
print("SPECTER2 loaded with proximity (retrieval) adapter")
print(f"Active adapters: {model.active_adapters}")
print("Distance metric: Euclidean (as per SPECTER2 training objective)")

#EMBEDDING FUNCTION
#this function takes a list of texts and returns their embeddings
#it processes them in batches so we dont run out of gpu memory
def embed_texts(texts, batch_size):
    all_embeddings = []
    
    #torch.no_grad turns off gradient tracking which we dont need for inference
    #it makes things faster and uses less memory
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            
            #tokenise the batch
            #max_length=512 is the limit specter2 was trained with
            inputs = tokenizer(batch, padding=True, truncation=True,
                               return_tensors="pt", return_token_type_ids=False,
                               max_length=512)
            
            #move the tokenised input onto the gpu
            inputs = {key: value.to(device) for key, value in inputs.items()}
            
            #run the model
            output = model(**inputs)
            
            #the embedding is the first token (CLS) of the last hidden state
            #this is how the specter2 docs say to get the paper embedding
            embeddings = output.last_hidden_state[:, 0, :]
            
            all_embeddings.append(embeddings)
            
            #simple progress print
            done = min(start + batch_size, len(texts))
            print(f"  embedded {done}/{len(texts)}", end="\r")
    
    print()
    #stick all the batches together into one big tensor
    return torch.cat(all_embeddings, dim=0)

#LOADING THE DATA
print("\nLoading data...")
neurips = pd.read_csv(neurips_path)
neurips = neurips.dropna(subset=["Abstract"]).reset_index(drop=True)
print(f"NeurIPS dataset: {len(neurips)} abstracts")

paraphrases = pd.read_csv(paraphrase_path)
print(f"Paraphrases: {len(paraphrases)}")

#EMBEDDING THE FULL DATASET
#abstract only, no title, to keep the comparison fair across all models
print("\nEmbedding the full NeurIPS dataset...")
dataset_abstracts = neurips["Abstract"].tolist()

start = time.time()
dataset_embeddings = embed_texts(dataset_abstracts, batch_size)
dataset_embedding_time = time.time() - start
print(f"Dataset embedding done in {dataset_embedding_time:.2f}s")

#EMBEDDING THE PARAPHRASES
print(f"\nEmbedding the {len(paraphrases)} paraphrases...")
paraphrase_texts = paraphrases["paraphrased_abstract"].tolist()

start = time.time()
paraphrase_embeddings = embed_texts(paraphrase_texts, batch_size)
paraphrase_embedding_time = time.time() - start
print(f"Paraphrase embedding done in {paraphrase_embedding_time:.2f}s")

#FINDING THE RANK OF THE CORRECT ORIGINAL
#for each paraphrase:
#1. compute the euclidean distance to every abstract in the dataset
#2. sort the dataset abstracts from smallest to largest distance
#   (smallest distance = most similar, because specter2 uses euclidean)
#3. find the position (rank) of the correct original abstract
print("\nCalculating ranks (using Euclidean distance)...")

#mapping each paper number to its row index in the neurips dataset
paper_number_to_index = {num: idx for idx, num in enumerate(neurips["Number"].values)}

ranks = []
skipped = []

for i in range(len(paraphrases)):
    correct_paper_number = paraphrases.iloc[i]["paper_number"]
    
    #if the correct paper isnt in the dataset we skip it
    if correct_paper_number not in paper_number_to_index:
        ranks.append(None)
        skipped.append(i)
        continue
    
    correct_index = paper_number_to_index[correct_paper_number]
    
    #torch.cdist computes the euclidean distance between the paraphrase
    #embedding and every dataset embedding
    #we add a dimension to the paraphrase so cdist sees it as a batch of 1
    query = paraphrase_embeddings[i].unsqueeze(0)
    distances = torch.cdist(query, dataset_embeddings)[0]
    
    #sorting ascending because for euclidean, smaller distance = more similar
    #so the most similar paper is at position 0
    ranked_indices = torch.argsort(distances, descending=False)
    
    #finding where the correct original landed, +1 so rank starts at 1
    rank = (ranked_indices == correct_index).nonzero()[0].item() + 1
    ranks.append(rank)

#SAVING DETAILED RESULTS (per-paper ranks plus the texts)
#we include the one-sentence paraphrase and the original abstract
#so high-rank cases can be inspected to understand why the model missed
#for example, if rank is 500 we can read both side by side and check
#whether the paraphrase lost too much meaning or the abstract is just
#close to many other papers in the dataset
results = pd.DataFrame({
    "experiment": paraphrases["experiment"].values,
    "paper_number": paraphrases["paper_number"].values,
    "title": paraphrases["title"].values,
    "rank_of_correct_original": ranks,
    "paraphrased_abstract": paraphrases["paraphrased_abstract"].values,
    "original_abstract": paraphrases["original_abstract"].values
})

results.to_csv(output_path, index=False)

#SAVING SUMMARY (one row appended to the shared all-models summary)
#calculates per-experiment averages, overall average, rank-1 count, timings
#each model adds its row to the same csv so all 3 sit in one comparison table
valid_results = results.dropna(subset=["rank_of_correct_original"])

#per-experiment averages
exp_averages = {}
for exp in sorted(valid_results["experiment"].unique()):
    exp_data = valid_results[valid_results["experiment"] == exp]
    exp_averages[f"avg_rank_exp_{exp}"] = exp_data["rank_of_correct_original"].mean()

overall_avg = valid_results["rank_of_correct_original"].mean()
rank_1_count = (valid_results["rank_of_correct_original"] == 1).sum()

#one row dataframe for this models results
new_row = pd.DataFrame([{
    "model": model_label,
    "similarity_metric": "euclidean",
    "n_paraphrases": len(valid_results),
    **exp_averages,
    "overall_avg_rank": overall_avg,
    "rank_1_hits": rank_1_count,
    "rank_1_percentage": rank_1_count / len(valid_results) * 100,
    "dataset_embedding_time_s": dataset_embedding_time,
    "paraphrase_embedding_time_s": paraphrase_embedding_time,
    "total_embedding_time_s": dataset_embedding_time + paraphrase_embedding_time,
}])

#append to the shared summary file
#if the file doesnt exist yet we create it
#if it does, we read it, remove any old row for this same model (lets us
#rerun cleanly without duplicating rows), then append the new row
if os.path.exists(shared_summary_path):
    existing = pd.read_csv(shared_summary_path)
    existing = existing[existing["model"] != model_label]
    summary = pd.concat([existing, new_row], ignore_index=True)
else:
    summary = new_row

summary.to_csv(shared_summary_path, index=False)

#PRINT SUMMARY
print(f"\n{'='*55}")
print(f"RESULTS: {model_label}")
print(f"{'='*55}")

print("\nAverage rank per experiment:")
for exp in sorted(valid_results["experiment"].unique()):
    exp_data = valid_results[valid_results["experiment"] == exp]
    print(f"  Experiment {exp}: average rank {exp_data['rank_of_correct_original'].mean():.2f}  (over {len(exp_data)} paraphrases)")

print(f"\nOverall average rank: {overall_avg:.2f}")
print(f"Times ranked correct original as #1: {rank_1_count}/{len(valid_results)}")

print(f"\nDataset embedding time: {dataset_embedding_time:.2f}s")
print(f"Paraphrase embedding time: {paraphrase_embedding_time:.2f}s")
print(f"Total embedding time: {dataset_embedding_time + paraphrase_embedding_time:.2f}s")

if skipped:
    print(f"\nWarning: {len(skipped)} paraphrases skipped (paper not found in dataset)")

print(f"\nDetailed results saved to {output_path}")
print(f"Summary row added to {shared_summary_path}")