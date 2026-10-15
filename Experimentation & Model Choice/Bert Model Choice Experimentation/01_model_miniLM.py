import os
import time
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path #so the paths work no matter where the script is run from

#duplicate detection experiment with all-MiniLM-L6-v2
#first of the 3 model scripts, one script per candidate model, all 3 built the same way
#the test itself is my own design, written up in section 5.3 of the dissertation

#what happens here in plain words:
#i have 250 abstracts that were rewritten as one plain english sentence each
#the rewrites keep the meaning but throw away all the technical words
#the model gets a rewritten sentence and has to find the original paper
#among all 5286 abstracts in the dataset
#if it really understands meaning it should point at the right paper
#if it only matches words it will get lost, since the words are all different
#the position where the correct paper lands is called its rank
#rank 1 means the model found it immediately, lower rank is better

#how the comparing works in one line: every text becomes a vector, a list of numbers,
#and cosine similarity scores how close two vectors point, higher score = closer meaning

#saves two csvs:
#results_minilm.csv keeps one row per paraphrase with its rank plus both texts,
#so high-rank misses can be opened later and read side by side
#summary_all_models.csv gets one row per model, all 3 scripts write into this same file,
#so the final comparison table is ready as soon as all 3 have been run

#libraries used in this script and where each one comes from:
#os, time and pathlib are python standard library, docs.python.org
#pandas reads and writes all the csvs, mckinney 2010, github.com/pandas-dev/pandas
#torch is pytorch, it does the tensor math and talks to the gpu,
#paszke et al. 2019, github.com/pytorch/pytorch
#sentence_transformers loads and runs the embedding model,
#its the official code of reimers & gurevych 2019 (sentence-bert),
#github.com/UKPLab/sentence-transformers, docs at sbert.net

#the model itself:
#all-MiniLM-L6-v2, built with the minilm distillation method of wang et al. 2020
#model card: huggingface.co/sentence-transformers/all-MiniLM-L6-v2
#used exactly as delivered, no finetuning, same rule for all 3 scripts

#MODEL SETUP
#two names on purpose. the long one is the models address on huggingface,
#thats what gets downloaded. the short one is only for my csvs and prints
model_name = "sentence-transformers/all-MiniLM-L6-v2" #full huggingface name, this is what gets downloaded
model_label = "all-MiniLM-L6-v2"  #shorter label used in the summary csv

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
OUTPUT_DIR = BASE_DIR / "01 MiniLM Result"
OUTPUT_DIR.mkdir(exist_ok=True)
output_path = OUTPUT_DIR / "results_minilm.csv"

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

#CHECKING GPU
#cuda is how pytorch talks to nvidia gpus
#must run on gpu, no fallback to cpu since that would skew timing comparison
#the whole point of the timing is comparing the 3 models on identical hardware
#a model that silently fell back to cpu would look much slower for reasons
#that have nothing to do with the model itself, so i stop instead
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Stopping. Fix the GPU setup before running this experiment.")

device = "cuda"
#print which card was found so the hardware is on record in the console output
print(f"CUDA available: True")
print(f"GPU: {torch.cuda.get_device_name(0)}")

#LOADING THE MODEL
#SentenceTransformer() downloads the model weights the first time and loads them,
#after that its cached locally so it starts fast. device puts it on the gpu
#i also print the metric and the vector size as a quick sanity check,
#minilm should say cosine and 384, if it says something else somethings off
model = SentenceTransformer(model_name, device=device)
print(f"\nModel loaded: {model_name}")
print(f"Similarity metric: {model.similarity_fn_name}")
print(f"Embedding dimensions: {model.get_embedding_dimension()}")

#LOADING THE DATA
#read_csv loads the file into a dataframe, basically a table living in memory
#dropna(subset=["Abstract"]) kicks out any row with an empty abstract,
#an empty text cant be embedded. my neurips file has none of those
#but the line costs nothing and protects the loop below
#reset_index renumbers the rows 0,1,2... after the drop, this matters later
#since the embeddings get stored in row order and the lookup dict counts on that
print("\nLoading data...")
neurips = pd.read_csv(neurips_path)
neurips = neurips.dropna(subset=["Abstract"]).reset_index(drop=True)
print(f"NeurIPS dataset: {len(neurips)} abstracts")

paraphrases = pd.read_csv(paraphrase_path)
print(f"Paraphrases: {len(paraphrases)}")

#EMBEDDING THE FULL DATASET
#embedding means turning each abstract into a vector of 384 numbers
#abstracts about the same topic end up with vectors that sit close together
#this is the background pool, every paraphrase gets compared against all of these

#tolist() hands encode a plain python list of the texts
#encode pushes them through the model in batches and returns one vector per text
#convert_to_tensor=True keeps the result on the gpu so the similarity step later is fast
#show_progress_bar just draws the little bar so i can see its alive
#timed with time.time() before and after, the difference is the embedding time
#this is the embed time number i report in table 5.2 of the dissertation
print("\nEmbedding the full NeurIPS dataset...")
start = time.time()
dataset_embeddings = model.encode(neurips["Abstract"].tolist(), convert_to_tensor=True,
                                  device=device, show_progress_bar=True)
dataset_embedding_time = time.time() - start
print(f"Dataset embedding done in {dataset_embedding_time:.2f}s")

#EMBEDDING THE PARAPHRASES
#same thing for the 250 one sentence rewrites, timed separately
#so both numbers can be looked at on their own
print("\nEmbedding the paraphrases...")
start = time.time()
paraphrase_embeddings = model.encode(paraphrases["paraphrased_abstract"].tolist(),
                                     convert_to_tensor=True, device=device,
                                     show_progress_bar=True)
paraphrase_embedding_time = time.time() - start
print(f"Paraphrase embedding done in {paraphrase_embedding_time:.2f}s")

#FINDING THE RANK OF THE CORRECT ORIGINAL
#this is the actual experiment
#for each paraphrase i compute its similarity to every single abstract,
#sort them from most similar to least similar, and write down at which
#position the correct original showed up
print("\nCalculating ranks...")

#the paraphrase csv stores the papers number, but the embeddings sit in row order
#this dict remembers for every paper number which row it lives in,
#so paper_number_to_index[some number] gives the row instantly, no searching every loop
paper_number_to_index = {num: idx for idx, num in enumerate(neurips["Number"].values)}

ranks = [] #one rank per paraphrase collects in here
skipped = [] #paraphrases i couldnt match to a dataset row end up here

for i in range(len(paraphrases)):
    #iloc[i] grabs row number i of the paraphrase table
    correct_paper_number = paraphrases.iloc[i]["paper_number"]
    
    #safety net, if a paraphrase points at a paper number thats not in the dataset
    #i skip it and count it instead of crashing 200 iterations in
    if correct_paper_number not in paper_number_to_index:
        ranks.append(None)
        skipped.append(i)
        continue
    
    correct_index = paper_number_to_index[correct_paper_number] #the dataset row where the correct original sits
    
    #model.similarity compares this one paraphrase vector against all 5286
    #dataset vectors in one go and returns one score per abstract
    #it uses the models own metric, for minilm thats cosine
    #the [0] is there since the result comes back wrapped in one extra dimension
    similarities = model.similarity(paraphrase_embeddings[i], dataset_embeddings)[0]
    
    #argsort with descending=True gives the dataset rows sorted by score,
    #best match first, so position 0 holds the most similar abstract
    ranked_indices = torch.argsort(similarities, descending=True)
    
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
#the loop takes each experiment number, keeps only its rows, averages the rank column
exp_averages = {}
for exp in sorted(valid_results["experiment"].unique()):
    exp_data = valid_results[valid_results["experiment"] == exp]
    exp_averages[f"avg_rank_exp_{exp}"] = exp_data["rank_of_correct_original"].mean()

overall_avg = valid_results["rank_of_correct_original"].mean() #average rank over all 250 together
rank_1_count = (valid_results["rank_of_correct_original"] == 1).sum() #true counts as 1, so the sum counts the rank-1 hits

#one row dataframe for this models results
#the ** takes the little dict of 5 experiment averages and spreads it out,
#each average becomes its own column in the row
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