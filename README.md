# From Abstracts to Rooms

**Poster-session assignment by semantic similarity and optimisation**
MSc Data Science dissertation, Lancaster University Leipzig (2026). Supervised by Dr. Pascal Welke. Graded 80%.
**Full dissertation:** [Master_Dissertation_Poster_Session__39991938.pdf](Master_Dissertation_Poster_Session__39991938.pdf) · **Poster:** [Enes Sejdini Dissertation Poster (1).pdf](Enes%20Sejdini%20Dissertation%20Poster%20%281%29.pdf)

Large conference poster sessions hold thousands of papers in rooms of different sizes, which makes placing related work together by hand impractical. This project is a tool that takes any conference's papers (a CSV with one abstract per paper) and a list of room sizes, and returns a seating plan in which every room holds papers on similar topics, every paper has a seat, no room is over capacity and no room is empty.

## How it works

| Stage | What happens | Why this choice |
|---|---|---|
| 1. Data | Accepted NeurIPS 2025 papers collected with the OpenReview API (5,286 papers) | Real conference at real scale |
| 2. Embedding | Each abstract becomes a 384-dimensional vector with `all-MiniLM-L6-v2` | Beat two scientific models on accuracy and speed (below) |
| 3. Clustering | Ward's hierarchical clustering, one group per ~10 papers (529 groups) | Every paper in exactly one group, no randomness, re-cut at any size almost for free |
| 4. Room assignment | GRASP packs whole groups into rooms under capacity and non-empty rules | Best average and smallest spread in a five-method benchmark |
| 5. Evaluation | Within-room paper similarity compared with a random assignment of the same room sizes | A similarity number only means something next to its random floor |

An exact CP-SAT model (Google OR-Tools) was tried first and ran for four hours without a proven result, which is why the assignment uses time-bounded heuristics.

## Results

**Embedding model selection.** 250 abstracts were rewritten as one plain-English sentence each (Gemma E2B, run locally with Ollama), and each model had to find the original paper among all 5,286. Lower rank is better.

| Model | Average rank | Rank-1 hits | Time to embed all papers |
|---|---|---|---|
| all-MiniLM-L6-v2 | **96.64** | **76 / 250** | **7.1 s** |
| SPECTER2 | 193.81 | 48 / 250 | 1,236.2 s |
| Aspire | 221.16 | 44 / 250 | 1,126.4 s |

The first version of this test was useless: every model found every paper at rank 1, because the paraphrases still contained rare technical terms, so the models were matching words rather than meaning. The paraphrase prompt was rewritten to ban technical terms, checked on a 25-query pilot, and only then was the full set of 250 generated and frozen.

**Room-assignment benchmark.** Five heuristics, 20 seeded attempts of 60 seconds each on the same 529 groups and 12 rooms.

| Method | Best objective | Average objective | Spread (std) |
|---|---|---|---|
| GRASP | 8,306.15 | **8,176.36** | **76.6** |
| Memetic algorithm | 8,312.05 | 8,150.35 | 110.2 |
| Matheuristic (CP-SAT inside a heuristic loop) | **8,326.37** | 7,893.15 | 302.6 |
| Variable Neighborhood Search | 8,286.03 | 7,610.40 | 329.6 |
| Tabu Search | 8,279.70 | 7,595.77 | 333.3 |

Friedman test: χ² = 49.855, p < 0.001. Holm-corrected pairwise Wilcoxon tests separate nine of ten pairs; GRASP and the Memetic algorithm cannot be separated (p = 0.355). GRASP was chosen for its higher average, smaller spread and simpler design.

**Final layouts.** The tool was run on NeurIPS 2025 and, once, on a 10,000-paper arXiv sample from nine fields that played no part in any design decision.

| Dataset | Papers | Rooms / seats | Within-room similarity | Random baseline | Lift |
|---|---|---|---|---|---|
| NeurIPS 2025 | 5,286 | 12 / 5,286 | 0.3395 | 0.3040 | +0.0355 (about 12%) |
| arXiv sample | 10,000 | 12 / 10,800 | 0.1735 | 0.0658 | +0.1077 (about 164%) |

Both plans seated every paper, kept every room within capacity and left no room empty. Three NeurIPS runs on the same machine returned the identical plan. On the arXiv run, rooms came out as single fields (finance, economics, biology, mathematics) even though the tool never reads the category labels, only the abstracts.

## Run it

```bash
pip install sentence-transformers torch scipy numpy pandas
cd "Full Application"
python main.py --papers path/to/papers.csv --rooms 700,600,550
```

- The papers file only needs an abstract column. Title, ID and category columns are optional and only make the output easier to read.
- Other column names: `--abstract-col "Summary"`, `--id-col`, `--category-col`. Output folder: `--out`.
- Started without flags, the tool asks for the file and the room sizes.
- Uses a GPU if one is available, otherwise the CPU.

Each run writes a new `Production N` folder with `layout.csv` (one row per paper and its room), `room_summary.csv` (fill, similarity and top categories per room) and `quality_report.txt` (similarity, random baseline and lift).

## Repository layout

| Folder | Contents |
|---|---|
| `Full Application/` | The tool: `main.py` plus one module each for embedding, clustering, GRASP and evaluation. `Production 1–3` are the NeurIPS runs, `Production 5` the arXiv run. |
| `Experimentation & Model Choice/` | Data collection (OpenReview, arXiv sample), paraphrase set, embedding-model comparison, clustering, the exact model and the five-heuristic benchmark, with all result CSVs. |

## Limitations

The tool reads only abstracts, and MiniLM reads the first 256 word pieces of each. Similarity is the model's judgement and has not yet been checked with conference visitors. On the arXiv run, the 800 spare seats gathered in two rooms. Search stops on the clock, so results are reproducible per machine.
