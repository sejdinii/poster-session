import time
import pandas as pd
import requests
from pathlib import Path #so the paths work no matter where the script is run from

#final paraphrasing script - full run
#250 papers, 50 per experiment across 5 experiments, no paper repeated
#uses the v2 prompt that bans technical vocabulary and produces one sentence
#overwrites paraphrased_abstracts.csv so the bert scripts need no changes

#ollama runs a local server on port 11434
ollama_url = "http://localhost:11434/api/generate"
model_name = "gemma4:e2b"

#anchor everything to this scripts own location instead of wherever the
#terminal happens to be standing.
#input: the neurips csv that dataset_API.py creates, in the sibling
#"Neurips Dataset" folder. if your dataset lives somewhere else, change
#only this input_path line
BASE_DIR = Path(__file__).resolve().parent
input_path = BASE_DIR.parent / "Neurips Dataset" / "Dataset" / "neurips_accepted_submissions.csv"

#output: a Summarized Dataset folder right next to this script, created
#automatically if its missing
OUTPUT_DIR = BASE_DIR / "Summarized Dataset"
OUTPUT_DIR.mkdir(exist_ok=True)
output_path = OUTPUT_DIR / "paraphrased_abstracts.csv"

#experiment setup - full run
n_experiments = 5
papers_per_experiment = 50
total_papers = n_experiments * papers_per_experiment  #250 total

#same seed so the 250 papers are consistent and reproducible
random_seed = 42

#-----LOADING THE DATA-----
#stop with a clear message if the dataset isnt where we expect it
if not input_path.exists():
    print(f"could not find the dataset csv here: {input_path}")
    print("run dataset_API.py first, or fix the input_path line at the top of this script.")
    raise SystemExit(1)

print("Loading neurips dataset...")
df = pd.read_csv(input_path)
print(f"Total papers: {len(df)}")

df = df.dropna(subset=["Abstract"])
print(f"Papers with valid abstracts: {len(df)}")

#-----SAMPLING 250 UNIQUE PAPERS-----
sampled = df.sample(n=total_papers, random_state=random_seed).reset_index(drop=True)
print(f"\nSelected {len(sampled)} unique papers total")

#assigning experiment numbers, first 50 -> exp 1, next 50 -> exp 2, etc
sampled["experiment"] = [(i // papers_per_experiment) + 1 for i in range(total_papers)]

print("Papers per experiment:")
print(sampled["experiment"].value_counts().sort_index())

#-----PARAPHRASING WITH LOCAL GEMMA-----
#aggressive prompt: ban technical terms, one sentence only, 14-year-old framing
prompt_template = """Rewrite the following scientific abstract as ONE single sentence using only plain everyday English. Imagine you are explaining this paper to a smart 14-year-old who has never read a scientific paper.

Strict rules:
- Do not reuse any technical terms, jargon, method names, model names, algorithm names, dataset names, acronyms or proper nouns from the original
- Replace any specialized vocabulary with plain descriptions that a non-expert reader would understand
- Use common everyday words throughout
- Write exactly ONE sentence, not multiple
- Capture what the paper does, why it matters, and what it found
- Return only the rewritten sentence with no introduction or explanation

Original abstract:
{abstract}"""

original_abstracts = []
paraphrased_abstracts = []
paraphrase_times = []
length_ratios = []
errors = []

print("\nStarting paraphrase generation with local Gemma (full 250 run)...")
print("First one will be slower because Gemma needs to load into memory")
print("This will take a while, let it run\n")

for i, row in sampled.iterrows():
    original = row["Abstract"]
    
    try:
        start = time.time()
        
        response = requests.post(
            ollama_url,
            json={
                "model": model_name,
                "prompt": prompt_template.format(abstract=original),
                "stream": False
            }
        )
        
        elapsed = time.time() - start
        
        if response.status_code != 200:
            raise Exception(f"Ollama returned status {response.status_code}: {response.text}")
        
        result = response.json()
        paraphrased = result["response"].strip()
        
        length_ratio = len(paraphrased) / len(original)
        
        original_abstracts.append(original)
        paraphrased_abstracts.append(paraphrased)
        paraphrase_times.append(elapsed)
        length_ratios.append(length_ratio)
        errors.append(None)
        
        print(f"  [{i+1}/{total_papers}] exp {row['experiment']} - done in {elapsed:.2f}s, length ratio: {length_ratio:.2f}")
    
    except Exception as e:
        print(f"  [{i+1}/{total_papers}] ERROR: {e}")
        original_abstracts.append(original)
        paraphrased_abstracts.append(None)
        paraphrase_times.append(None)
        length_ratios.append(None)
        errors.append(str(e))

#-----SAVING RESULTS-----
results = pd.DataFrame({
    "experiment": sampled["experiment"].values,
    "paper_number": sampled["Number"].values,
    "title": sampled["Title"].values,
    "original_abstract": original_abstracts,
    "paraphrased_abstract": paraphrased_abstracts,
    "paraphrase_time_seconds": paraphrase_times,
    "length_ratio": length_ratios,
    "error": errors
})

results.to_csv(output_path, index=False)

#-----SUMMARY-----
successful = results[results["paraphrased_abstract"].notna()]
failed = results[results["paraphrased_abstract"].isna()]

print(f"\n{'='*50}")
print("PARAPHRASE GENERATION COMPLETE (full 250 run)")
print(f"{'='*50}")
print(f"Total attempted: {len(results)}")
print(f"Successful: {len(successful)}")
print(f"Failed: {len(failed)}")

if len(successful) > 0:
    print(f"\nTotal time: {successful['paraphrase_time_seconds'].sum():.2f} seconds")
    print(f"Average time per paraphrase: {successful['paraphrase_time_seconds'].mean():.2f} seconds")
    print(f"Average length ratio: {successful['length_ratio'].mean():.2f}")
    print(f"Length ratio range: {successful['length_ratio'].min():.2f} - {successful['length_ratio'].max():.2f}")

print("\nSuccessful paraphrases per experiment:")
print(successful["experiment"].value_counts().sort_index())

print(f"\nSaved to {output_path}")