import os
import time
import pandas as pd
import torch
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from pathlib import Path #so the paths work no matter where the script is run from

#duplicate detection experiment for SPECTER2
#second of the 3 model scripts, same test as the minilm one, described in section 5.3 of the dissertation

#what happens here in plain words:
#i have 250 abstracts that were rewritten as one plain english sentence each
#the model gets a rewritten sentence and has to find the original paper
#among all 5286 abstracts in the dataset
#the position where the correct paper lands is called its rank
#rank 1 means the model found it immediately, lower rank is better

#SPECTER2 differs from all-MiniLM-L6-v2 in two important ways:

#1. loading: SPECTER2 does not load as a simple SentenceTransformer
#it loads as a base model (allenai/specter2_base) plus a separate
#retrieval adapter (allenai/specter2, the proximity adapter)
#i embed by tokenising the text, running the model, and taking
#the first token (CLS) of the last hidden state as the embedding
#thats the recipe from the specter2 model card

#2. distance metric: SPECTER2 was trained with a triplet margin loss
#based on EUCLIDEAN distance, not cosine
#so i rank papers by euclidean distance (smaller distance = more similar)
#this is confirmed from the specter2 paper (singh et al. 2023) and the ai2 blog
#judging each model by its own metric is the rule set in section 5.3

#everything else is the same as the minilm experiment:
#reads however many paraphrases are in the csv, ranks against the full dataset

#saves two csvs:
#results_specter2.csv keeps one row per paraphrase with its rank plus both texts,
#so high-rank misses can be opened later and read side by side
#summary_all_models.csv gets one row per model, all 3 scripts write into this same file,
#so the final comparison table is ready as soon as all 3 have been run

#note: i use abstract only (no title) for all models to keep the comparison fair
#the specter2 docs actually suggest title plus abstract, but giving one model
#more text than the others would break the comparison, section 5.3 explains this

#libraries used in this script and where each one comes from:
#os, time and pathlib are python standard library, docs.python.org
#pandas reads and writes the csvs, mckinney 2010, github.com/pandas-dev/pandas
#torch is pytorch, tensor math and the gpu work, paszke et al. 2019, github.com/pytorch/pytorch
#transformers gives the tokenizer, wolf et al. 2020, github.com/huggingface/transformers
#adapters loads the specter2 adapter, poth et al. 2023, github.com/adapter-hub/adapters

#the model itself:
#specter2 (singh et al. 2023), successor of specter (cohan et al. 2020)
#official repo: github.com/allenai/SPECTER2
#model cards: huggingface.co/allenai/specter2_base and huggingface.co/allenai/specter2
#described in sections 2.2 and 5.1 of the dissertation, results feed table 5.2
#used exactly as delivered, no finetuning, same rule for all 3 scripts

#MODEL SETUP
#two names since specter2 comes in two pieces, the base model and the adapter
#that sits on top of it, both get downloaded from huggingface
base_model_name = "allenai/specter2_base"
adapter_name = "allenai/specter2"  #the proximity / retrieval adapter
model_label = "SPECTER2"  #label used in the shared summary csv

#PATHS
#anchor everything to this scripts own location instead of wherever the
#terminal happens to be standing. the two inputs live in sibling folders,
#the outputs land in folders created right next to this script.
#if an input lives somewhere else, change only its line here

#how the anchoring works: __file__ is this scripts own file,
#resolve() turns that into a full path, and parent is the folder its sitting in
BASE_DIR = Path(__file__).resolve().parent   #...\Bert Model Choice Experimentation

#the / glues folder names onto a path, .parent first steps one folder up
#the dataset csv is made by dataset_API.py, the paraphrases by summarization_API.py
neurips_path = BASE_DIR.parent / "Neurips Dataset" / "Dataset" / "neurips_accepted_submissions.csv"
paraphrase_path = BASE_DIR.parent / "Neurips Dataset (Summarized)" / "Summarized Dataset" / "paraphrased_abstracts.csv"

#this models own result folder, created automatically if its missing
#exist_ok=True means if the folder already exists just carry on, dont crash
OUTPUT_DIR = BASE_DIR / "02 Specter2 Result"
OUTPUT_DIR.mkdir(exist_ok=True)
output_path = OUTPUT_DIR / "results_specter2.csv"

#the shared summary gets its own folder too - all 3 model scripts write
#into the same csv there, one row each
SUMMARY_DIR = BASE_DIR / "Summary of Results"
SUMMARY_DIR.mkdir(exist_ok=True)
shared_summary_path = SUMMARY_DIR / "summary_all_models.csv"

#stop early with a clear message if an input file isnt where i expect it
#way better than a confusing pandas crash halfway down the script
#exists() simply asks the file system if theres a file at that path
#SystemExit(1) ends the script, the 1 is the standard code for ended with a problem
if not neurips_path.exists():
    print(f"could not find the neurips dataset here: {neurips_path}")
    print("run dataset_API.py first, or fix the neurips_path line at the top of this script.")
    raise SystemExit(1)
if not paraphrase_path.exists():
    print(f"could not find the paraphrases here: {paraphrase_path}")
    print("run summarization_API.py first, or fix the paraphrase_path line at the top of this script.")
    raise SystemExit(1)

#how many abstracts i push through the model at once when embedding
#smaller batch since specter2 is a bigger model and my gpu has 6gb
#minilm handled batching on its own inside encode(), here i do it by hand below
batch_size = 16

#CHECKING GPU
#cuda is how pytorch talks to nvidia gpus
#this experiment must run on the gpu
#if cuda is not available i stop instead of falling back to cpu
#because cpu timings would make the compute comparison meaningless
#the whole point of the timing is all 3 models on identical hardware
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Stopping. Fix the GPU setup before running this experiment.")

device = "cuda"
#print which card was found so the hardware is on record in the console output
print(f"CUDA available: True")
print(f"GPU: {torch.cuda.get_device_name(0)}")

#LOADING THE MODEL
#three steps here instead of minilms one line
#first the tokenizer, then the base model, then the adapter gets attached on top
#the tokenizer chops text into the word pieces the model actually reads
print("\nLoading SPECTER2...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoAdapterModel.from_pretrained(base_model_name)

#an adapter is a small extra layer trained for one specific job,
#this one (proximity) is trained for finding related papers
#load_adapter downloads it from huggingface (source="hf"),
#load_as gives the adapter a name, set_active makes it the one being used
model.load_adapter(adapter_name, source="hf", load_as="proximity", set_active=True)

#explicitly activate the adapter for the forward pass
#the set_active=True above does not always stick on its own
#so i set it directly here to make sure the adapter is actually used
#do not remove this line, without it the model can silently run bare
#and the results would be wrong while looking perfectly fine
model.set_active_adapters("proximity")

#move the model onto the gpu and set it to evaluation mode
#eval mode turns off training-only behaviour like dropout
model.to(device)
model.eval()
#the prints put on record what actually got loaded and which metric applies
print("SPECTER2 loaded with proximity (retrieval) adapter")
print(f"Active adapters: {model.active_adapters}")
print("Distance metric: Euclidean (as per SPECTER2 training objective)")

#EMBEDDING FUNCTION
#minilm did all of this inside encode(), specter2 has no helper like that
#so this function does the job by hand
#it takes a list of texts and returns their embeddings,
#processed in batches so i dont run out of gpu memory
def embed_texts(texts, batch_size):
    all_embeddings = [] #the finished batches collect in here
    
    #torch.no_grad turns off gradient tracking which i dont need for inference
    #gradients are only for training, skipping them is faster and uses less memory
    with torch.no_grad():
        #range(0, len, batch_size) walks the list in jumps of one batch
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size] #slice out the next batch of texts
            
            #tokenise the batch, meaning chop the texts into model-readable pieces
            #padding=True makes every row the same length, truncation cuts too-long ones
            #max_length=512 is the window specter2 was trained with, per the model card
            #return_tensors="pt" gives pytorch tensors back
            inputs = tokenizer(batch, padding=True, truncation=True,
                               return_tensors="pt", return_token_type_ids=False,
                               max_length=512)
            
            #move the tokenised input onto the gpu
            #the model is already there and both must sit on the same device
            inputs = {key: value.to(device) for key, value in inputs.items()}
            
            #run the model, the ** spreads the dict out into the models arguments
            output = model(**inputs)
            
            #the embedding is the first token (CLS) of the last hidden state
            #this is how the specter2 docs say to get the paper embedding
            #the cls token idea itself comes from bert (devlin et al. 2019)
            #[:, 0, :] means every text in the batch, first token, all 768 numbers
            embeddings = output.last_hidden_state[:, 0, :]
            
            all_embeddings.append(embeddings)
            
            #simple progress print, end="\r" rewrites the same line instead of spamming
            done = min(start + batch_size, len(texts))
            print(f"  embedded {done}/{len(texts)}", end="\r")
    
    print() #jump to a fresh line once the progress line is finished
    #stick all the batches together into one big tensor, one row per text
    return torch.cat(all_embeddings, dim=0)

#LOADING THE DATA
#read_csv loads the file into a dataframe, basically a table living in memory
#dropna kicks out any row with an empty abstract, an empty text cant be embedded
#my neurips file has none of those but the line costs nothing
#reset_index renumbers the rows 0,1,2... so the embedding row order
#and the lookup dict further down stay in step
print("\nLoading data...")
neurips = pd.read_csv(neurips_path)
neurips = neurips.dropna(subset=["Abstract"]).reset_index(drop=True)
print(f"NeurIPS dataset: {len(neurips)} abstracts")

paraphrases = pd.read_csv(paraphrase_path)
print(f"Paraphrases: {len(paraphrases)}")

#EMBEDDING THE FULL DATASET
#the background pool, every paraphrase gets compared against all of these
#abstract only, no title, to keep the comparison fair across all models (section 5.3)
#timed with time.time() before and after, the difference is the embed time
#this number feeds table 5.2 of the dissertation
print("\nEmbedding the full NeurIPS dataset...")
dataset_abstracts = neurips["Abstract"].tolist() #plain python list of the texts

start = time.time()
dataset_embeddings = embed_texts(dataset_abstracts, batch_size)
dataset_embedding_time = time.time() - start
print(f"Dataset embedding done in {dataset_embedding_time:.2f}s")

#EMBEDDING THE PARAPHRASES
#same thing for the 250 one sentence rewrites, timed separately
#so both numbers can be looked at on their own
print(f"\nEmbedding the {len(paraphrases)} paraphrases...")
paraphrase_texts = paraphrases["paraphrased_abstract"].tolist()

start = time.time()
paraphrase_embeddings = embed_texts(paraphrase_texts, batch_size)
paraphrase_embedding_time = time.time() - start
print(f"Paraphrase embedding done in {paraphrase_embedding_time:.2f}s")

#FINDING THE RANK OF THE CORRECT ORIGINAL
#this is the actual experiment
#for each paraphrase:
#1. compute the euclidean distance to every abstract in the dataset
#2. sort the dataset abstracts from smallest to largest distance
#(smallest distance = most similar, because specter2 uses euclidean)
#3. find the position (rank) of the correct original abstract
print("\nCalculating ranks (using Euclidean distance)...")

#the paraphrase csv stores the papers number, but the embeddings sit in row order
#this dict remembers for every paper number which row it lives in,
#so the lookup is instant instead of searching the whole dataframe every loop
paper_number_to_index = {num: idx for idx, num in enumerate(neurips["Number"].values)}

ranks = [] #one rank per paraphrase collects in here
skipped = [] #paraphrases i couldnt match to a dataset row end up here

for i in range(len(paraphrases)):
    #iloc[i] grabs row number i of the paraphrase table
    correct_paper_number = paraphrases.iloc[i]["paper_number"]
    
    #safety net, if the correct paper isnt in the dataset i skip it and count it
    #instead of crashing 200 iterations in
    if correct_paper_number not in paper_number_to_index:
        ranks.append(None)
        skipped.append(i)
        continue
    
    correct_index = paper_number_to_index[correct_paper_number] #the dataset row where the correct original sits
    
    #torch.cdist computes the euclidean distance between the paraphrase
    #embedding and every dataset embedding in one go, 5286 distances at once
    #unsqueeze(0) adds a dimension so cdist sees the paraphrase as a batch of 1
    #the [0] then unwraps that batch dimension again on the way out
    query = paraphrase_embeddings[i].unsqueeze(0)
    distances = torch.cdist(query, dataset_embeddings)[0]
    
    #sorting ascending because for euclidean, smaller distance = more similar
    #so the most similar paper is at position 0
    #note this is the opposite direction of the minilm script, which sorts descending
    ranked_indices = torch.argsort(distances, descending=False)
    
    #ranked_indices == correct_index is true only at the spot where the correct paper sits
    #nonzero returns that spot, item() turns it into a normal python number
    #+1 so rank starts at 1 like a person would count, rank 1 = found it immediately
    rank = (ranked_indices == correct_index).nonzero()[0].item() + 1
    ranks.append(rank)

#SAVING DETAILED RESULTS (per-paper ranks plus the texts)
#the DataFrame is built column by column, one row per paraphrase
#i include the one-sentence paraphrase and the original abstract
#so high-rank cases can be inspected to understand why the model missed
#for example, if rank is 500 i can read both side by side and check
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

results.to_csv(output_path, index=False) #index=False so row numbers dont get written as an extra column

#SAVING SUMMARY (one row appended to the shared all-models summary)
#this is the row that ends up feeding table 5.2 of the dissertation
#dropna throws away the skipped ones so the averages arent polluted by empty ranks
valid_results = results.dropna(subset=["rank_of_correct_original"])

#per-experiment averages
#the 250 papers are split into 5 experiments of 50, i average each one separately
#so i can see if a model is consistently good or just got lucky on one batch,
#section 5.3 of the dissertation explains why the split is there
exp_averages = {}
for exp in sorted(valid_results["experiment"].unique()):
    exp_data = valid_results[valid_results["experiment"] == exp]
    exp_averages[f"avg_rank_exp_{exp}"] = exp_data["rank_of_correct_original"].mean()

overall_avg = valid_results["rank_of_correct_original"].mean() #average rank over all 250 together
rank_1_count = (valid_results["rank_of_correct_original"] == 1).sum() #true counts as 1, so the sum counts the rank-1 hits

#one row dataframe for this models results
#the ** takes the little dict of 5 experiment averages and spreads it out,
#each average becomes its own column in the row
#similarity_metric is written as euclidean by hand since thats how this model ranks
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
#if the file doesnt exist yet i create it
#if it does, i read it, remove any old row for this same model (lets me
#rerun cleanly without duplicating rows), then append the new row
#ignore_index=True renumbers the rows after gluing the two tables together
if os.path.exists(shared_summary_path):
    existing = pd.read_csv(shared_summary_path)
    existing = existing[existing["model"] != model_label]
    summary = pd.concat([existing, new_row], ignore_index=True)
else:
    summary = new_row

summary.to_csv(shared_summary_path, index=False)

#PRINT SUMMARY
#everything below only prints what was already saved, nothing new is computed here
#the ='*55 bit just prints a row of equal signs as a divider in the console
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

#only shows up if something got skipped, on a clean run this stays silent
if skipped:
    print(f"\nWarning: {len(skipped)} paraphrases skipped (paper not found in dataset)")

print(f"\nDetailed results saved to {output_path}")
print(f"Summary row added to {shared_summary_path}")