import os
import time
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path #so the paths work no matter where the script is run from

#duplicate detection experiment with all-MiniLM-L6-v2
#reads whatever paraphrases are in the csv (250 for the full run)
#for each paraphrase, ranks all dataset abstracts by cosine similarity
#and finds the position of the correct original
#
#saves two csvs:
# - results_minilm.csv: per-paper ranks plus the paraphrase and original text
#   so high-rank misses can be inspected to see why the model missed
# - summary_all_models.csv: one row per model, all 3 models share this file
#   so the final comparison table is ready as soon as all 3 have been run

#MODEL SETUP
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_label = "all-MiniLM-L6-v2"  #shorter label used in the summary csv

#PATHS
#anchor everything to this scripts own location instead of wherever the
#terminal happens to be standing. the two inputs live in sibling folders,
#the outputs land in folders created right next to this script.
#if an input lives somewhere else, change only its line here
BASE_DIR = Path(__file__).resolve().parent   #...\Bert Model Choice Experimentation

neurips_path = BASE_DIR.parent / "Neurips Dataset" / "Dataset" / "neurips_accepted_submissions.csv"
paraphrase_path = BASE_DIR.parent / "Neurips Dataset (Summarized)" / "Summarized Dataset" / "paraphrased_abstracts.csv"

#this models own result folder, created automatically if its missing
OUTPUT_DIR = BASE_DIR / "01 MiniLM Result"
OUTPUT_DIR.mkdir(exist_ok=True)
output_path = OUTPUT_DIR / "results_minilm.csv"

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

#CHECKING GPU
#must run on gpu, no fallback to cpu since that would skew timing comparison
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Stopping. Fix the GPU setup before running this experiment.")

device = "cuda"
print(f"CUDA available: True")
print(f"GPU: {torch.cuda.get_device_name(0)}")

#LOADING THE MODEL
model = SentenceTransformer(model_name, device=device)
print(f"\nModel loaded: {model_name}")
print(f"Similarity metric: {model.similarity_fn_name}")
print(f"Embedding dimensions: {model.get_embedding_dimension()}")

#LOADING THE DATA
print("\nLoading data...")
neurips = pd.read_csv(neurips_path)
neurips = neurips.dropna(subset=["Abstract"]).reset_index(drop=True)
print(f"NeurIPS dataset: {len(neurips)} abstracts")

paraphrases = pd.read_csv(paraphrase_path)
print(f"Paraphrases: {len(paraphrases)}")

#EMBEDDING THE FULL DATASET
#the background pool, every paraphrase gets compared against these
#timed for the compute metric
print("\nEmbedding the full NeurIPS dataset...")
start = time.time()
dataset_embeddings = model.encode(neurips["Abstract"].tolist(), convert_to_tensor=True,
                                  device=device, show_progress_bar=True)
dataset_embedding_time = time.time() - start
print(f"Dataset embedding done in {dataset_embedding_time:.2f}s")

#EMBEDDING THE PARAPHRASES
print("\nEmbedding the paraphrases...")
start = time.time()
paraphrase_embeddings = model.encode(paraphrases["paraphrased_abstract"].tolist(),
                                     convert_to_tensor=True, device=device,
                                     show_progress_bar=True)
paraphrase_embedding_time = time.time() - start
print(f"Paraphrase embedding done in {paraphrase_embedding_time:.2f}s")

#FINDING THE RANK OF THE CORRECT ORIGINAL
#for each paraphrase, compute cosine similarity vs every dataset abstract
#sort to find where the correct original landed
print("\nCalculating ranks...")

paper_number_to_index = {num: idx for idx, num in enumerate(neurips["Number"].values)}

ranks = []
skipped = []

for i in range(len(paraphrases)):
    correct_paper_number = paraphrases.iloc[i]["paper_number"]
    
    if correct_paper_number not in paper_number_to_index:
        ranks.append(None)
        skipped.append(i)
        continue
    
    correct_index = paper_number_to_index[correct_paper_number]
    
    #cosine similarity (model.similarity uses the models own metric)
    similarities = model.similarity(paraphrase_embeddings[i], dataset_embeddings)[0]
    
    #descending sort, highest similarity at position 0
    ranked_indices = torch.argsort(similarities, descending=True)
    
    #+1 so rank starts at 1
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
    "similarity_metric": str(model.similarity_fn_name),
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