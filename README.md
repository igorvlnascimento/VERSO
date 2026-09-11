# VERSO — VERbalization and Structural mOdularization

VERSO is a modular pipeline for **complex ontology matching** (OM): given two OWL/RDF ontologies, it finds semantic correspondences between their concepts and expresses them in [EDOAL](http://alignapi.gforge.inria.fr/edoal.html) format. The pipeline chains six stages — Preprocessing → Verbalization → Candidate Generation → Matching → Postprocessing → Evaluation — and is driven by a YAML config file via [Hydra](https://hydra.cc).

---

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Downloading the Benchmark Datasets](#downloading-the-benchmark-datasets)
4. [Quick Start](#quick-start)
5. [Understanding the Pipeline](#understanding-the-pipeline)
   - [Stage 1 — Preprocessing](#stage-1--preprocessing)
   - [Stage 2 — Verbalization](#stage-2--verbalization)
   - [Stage 3 — Candidate Generation](#stage-3--candidate-generation)
   - [Stage 4 — Matching](#stage-4--matching)
   - [Stage 5 — Postprocessing](#stage-5--postprocessing)
   - [Stage 6 — Evaluation](#stage-6--evaluation)
6. [Configuration Reference](#configuration-reference)
7. [Choosing a Backend](#choosing-a-backend)
8. [Outputs and Experiment Tracking](#outputs-and-experiment-tracking)
9. [Adding Your Own Ontology Pair](#adding-your-own-ontology-pair)
10. [Project Structure](#project-structure)

---

## Requirements

- Python ≥ 3.10
- A CUDA-capable GPU (recommended for the default `vllm`/`transformers` backends)
- `wget`, `curl` (for the benchmark downloader)

The Python dependencies are declared in `pyproject.toml`:

```
hydra-core, mlflow, networkx, rdflib, transformers, vllm,
llama-cpp-python, lxml, bs4
```

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd OM_System

# Create and activate a virtual environment (uv recommended)
uv venv && source .venv/bin/activate

# Install the project and all dependencies
uv pip install -e .
```

---

## Downloading the Benchmark Datasets

The system ships with a download script that fetches OWL files and reference alignments from the [complex-OM-benchmark](https://github.com/liseda-lab/complex-OM-benchmark) repository.

```bash
chmod +x download_benchmark.sh
./download_benchmark.sh
```

This populates:

```
datasets/
  conference/   cmt.owl  conference.owl  confOf.owl  edas.owl  ekaw.owl
  enslaved/     enslaved.owl  wikidata.owl
  geolink/      gbo.owl  gmo.owl
  hydrography/  cree.owl  hydro3.owl  hydrOntology.owl  swo.owl
  taxon/        agrovoc.owl  dbpedia.owl  taxon.owl  taxref.owl

references/
  conference/   cmt-confOf.owl  cmt-conference.owl  ...
  enslaved/     enslaved-wikidata.owl
  geolink/      gbo-gmo.owl
  hydrography/  cree-swo.owl  hydro3-swo.owl  ...
  taxon/        taxon-agrovoc.edoal  taxon-dbpedia.edoal  ...
```

The `conference` dataset is fetched from population level `popconference-0-v1` by default. Edit the `CONFERENCE_VERSION` variable at the top of the script to use a different level (`0`, `20`, `40`, `60`, `80`, `100`).

---

## Quick Start

Run the pipeline on the default pair (`cmt` → `conference` from the `conference` dataset):

```bash
python -m stage.pipeline
```

Override any config key on the command line using Hydra's dot-notation:

```bash
# Different ontology pair
python -m stage.pipeline \
    dataset_name=geolink \
    ontology_source_name=gbo \
    ontology_target_name=gmo

# Use a lightweight CPU-friendly backend
python -m stage.pipeline \
    candidate_generation.backend=ollama \
    matching.backend=ollama \
    matching.llm_model=llama3
```

Results (precision, recall, F1) are printed to stdout and logged to MLflow.

---

## Understanding the Pipeline

```
OWL files
   │
   ▼
┌──────────────┐
│ Preprocessing│  Parse ontologies → NetworkX graphs → modularize
└──────┬───────┘
       │  modules (lists of RDF triples)
       ▼
┌──────────────┐
│ Verbalization│  Convert IRI triples → human-readable text
└──────┬───────┘
       │  verbalized text modules
       ▼
┌──────────────────────┐
│ Candidate Generation │  Embed modules → cosine similarity → top-k candidates
└──────┬───────────────┘
       │  (source_module, target_module) pairs
       ▼
┌──────────┐
│ Matching │  LLM generates EDOAL alignment from each pair
└──────┬───┘
       │  raw LLM output strings
       ▼
┌───────────────┐
│ Postprocessing│  Parse + merge + repair EDOAL XML fragments
└──────┬────────┘
       │  final EDOAL file
       ▼
┌────────────┐
│ Evaluation │  Compare against reference → precision / recall / F1
└────────────┘
```

### Stage 1 — Preprocessing

**Code:** `stage/preprocessing/`

The source ontology is divided into overlapping *modules* — small, semantically coherent subgraphs. The target ontology is represented as a random sample of its graph. Both are stored as lists of `(subject, relation, object)` triples.

**Modularizer types** (set via `preprocessing.modularizer_type`):

| Value | Class | Description |
|-------|-------|-------------|
| `pagerank` | `PageRankModularizer` | Picks the top-N nodes by global PageRank as module centres |
| `ppr` | `PPRModularizer` | Runs Personalised PageRank from each centre to expand the group |

Each centre is expanded to a local subgraph via BFS (`BFSExpander`), up to `max_length_subgraph` triples.

Key parameters:
- `max_length_pagerank` — number of modules to generate from the source ontology (default 15)
- `depth` — BFS depth when expanding a module centre
- `max_length_subgraph` — max triples per module
- `max_length_target_sample` — how many nodes to sample from the target ontology
- `relations_to_remove` — OWL relations to strip (e.g. `disjointWith`)

Intermediate module files are written to `outputs/preprocessing/`.

---

### Stage 2 — Verbalization

**Code:** `stage/verbalization/`

Verbalization converts raw IRIs in the triples into readable text strings before passing them to embedding/LLM models.

Three verbalizer types are available (set independently for candidate generation and matching via `verbalizer_type.candidate_generator` and `verbalizer_type.matcher`):

| Value | Class | What it does |
|-------|-------|--------------|
| `base` | `BaseVerbalizer` | Keeps the full IRI as-is |
| `label` | `LabelVerbalizer` | Extracts the human-readable label from the IRI fragment |
| `natural` | `NaturalVerbalizer` | Applies natural language rules on top of labels: `subClassOf` → `is_a`, `equivalentClass` → `is_equivalent_to`, camelCase → `snake_case`, etc. |

Example — for a triple `(cmt:Paper, subClassOf, cmt:Submission)`:

| Verbalizer | Output |
|------------|--------|
| `base` | `(http://…/cmt#Paper, subClassOf, http://…/cmt#Submission)` |
| `label` | `(Paper, subClassOf, Submission)` |
| `natural` | `(Paper, is_a, Submission)` |

---

### Stage 3 — Candidate Generation

**Code:** `stage/candidate_generation/`

An embedding model encodes every verbalized source module and every verbalized target module into dense vectors. Cosine similarity is then computed between all source–target pairs, and the top-`max_similarities` target modules are kept for each source module.

Those top-k target nodes are expanded into full target subgraphs (again via BFS), producing `(source_module, target_module)` pairs that are fed to the matching stage.

Key parameters:
- `embedding_model_name` — any HuggingFace-compatible embedding model (default `Qwen/Qwen3-Embedding-8B`)
- `backend` — `transformers` | `ollama` | `llamacpp`
- `batch_size` — batch size for encoding
- `max_length` — token limit for the encoder
- `max_similarities` — how many top-k target candidates to keep per source module

Outputs written to `outputs/candidate_generation/candidates.json` and `entities_similarities.json`.

---

### Stage 4 — Matching

**Code:** `stage/matching/`

For each `(source_module, target_module)` pair the system builds a prompt and sends it to an LLM, asking it to produce an EDOAL alignment file for the two ontology fragments. The prompt includes two few-shot examples (read from `samples/sample1_<verbalizer>.txt` and `samples/sample2_<verbalizer>.txt`).

Key parameters:
- `llm_model` — model name (default `Qwen/Qwen3-14B`)
- `backend` — `vllm` | `ollama` | `llamacpp`
- `max_model_len` / `max_num_batched_tokens` — context length controls for vLLM
- `k` — number of samples to generate per prompt
- `temperature`, `top_p`, `top_k` — sampling parameters

Outputs written to `outputs/matching/prompts.json` and `results.json`.

---

### Stage 5 — Postprocessing

**Code:** `stage/postprocessing/`

The raw LLM outputs are XML fragments. Postprocessing:
1. Extracts the EDOAL fragment from each response (`AnswerExtractor`)
2. Repairs truncated or malformed XML
3. Merges all fragments into a single EDOAL document
4. Resolves abbreviated concept names back to full IRIs using the concept dictionaries built by the verbalizers

The final alignment file is written to `outputs/postprocessing/<source>-<target>.txt`.

---

### Stage 6 — Evaluation

**Code:** `stage/evaluation/`

The predicted EDOAL document is compared against the reference alignment (`references/<dataset>/<source>-<target>.owl`). The evaluator computes standard information-retrieval metrics:

- **Precision** — fraction of predicted correspondences that are correct
- **Recall** — fraction of reference correspondences that were predicted
- **F1** — harmonic mean of precision and recall

All three metrics are logged to MLflow and returned by `pipeline.run()`.

---

## Configuration Reference

The main config file is `stage/config.yaml`. All values can be overridden on the command line.

```yaml
# ── Input ────────────────────────────────────────────────────────────
dataset_name: "conference"          # folder inside datasets/
ontology_source_name: "cmt"         # file stem (looks for .owl then .rdf)
ontology_target_name: "conference"
seeds: 1420172                      # random seed

# ── Verbalization ────────────────────────────────────────────────────
verbalizer_type:
  candidate_generator: "base"       # base | label | natural
  matcher: "base"

# ── Preprocessing ────────────────────────────────────────────────────
preprocessing:
  modularizer_type: "pagerank"      # pagerank | ppr
  max_length_pagerank: 15           # number of source modules
  depth: 1                          # BFS depth
  max_length_subgraph: 20           # max triples per module
  max_length_target_sample: 3000    # target sample size
  relations_to_remove: ["disjointWith"]

# ── Candidate Generation ─────────────────────────────────────────────
candidate_generation:
  embedding_model_name: "Qwen/Qwen3-Embedding-8B"
  backend: "transformers"           # transformers | ollama | llamacpp
  batch_size: 2
  max_length: 16384
  max_similarities: 5               # top-k candidates per source module
  max_length_similarities: 20       # BFS expansion of chosen candidates

# ── Matching ─────────────────────────────────────────────────────────
matching:
  llm_model: "Qwen/Qwen3-14B"
  backend: "vllm"                   # vllm | ollama | llamacpp
  dtype: "float16"
  max_model_len: 16384
  temperature: 1.0
  top_p: 1.0
  k: 1                              # completions per prompt
```

---

## Choosing a Backend

| Scenario | Embedding backend | LLM backend | Notes |
|----------|-------------------|-------------|-------|
| Multi-GPU server | `transformers` | `vllm` | Default; highest throughput |
| Single consumer GPU | `transformers` | `vllm` | Reduce `max_model_len` if OOM |
| CPU / no GPU | `ollama` | `ollama` | Requires Ollama running locally |
| GGUF model file | `llamacpp` | `llamacpp` | Pass the `.gguf` path as model name |

**Ollama example:**
```bash
# Start Ollama in another terminal
ollama serve

python -m stage.pipeline \
    candidate_generation.backend=ollama \
    candidate_generation.embedding_model_name=nomic-embed-text \
    matching.backend=ollama \
    matching.llm_model=llama3
```

**llama.cpp example:**
```bash
python -m stage.pipeline \
    candidate_generation.backend=llamacpp \
    candidate_generation.embedding_model_name=/path/to/embed.gguf \
    matching.backend=llamacpp \
    matching.llm_model=/path/to/model.gguf \
    matching.n_gpu_layers=35
```

---

## Outputs and Experiment Tracking

Every run writes intermediate and final artifacts under `outputs/`:

```
outputs/
  preprocessing/
    source_modules.json       # source ontology modules
    target_modules.json       # target ontology sample modules
  candidate_generation/
    candidates.json           # top-k target candidates per source module
    entities_similarities.json
  matching/
    prompts.json              # prompts sent to the LLM
    results.json              # raw LLM responses
  postprocessing/
    <source>-<target>.txt     # final EDOAL alignment
```

Metrics and artifacts are also tracked in **MLflow**. The tracking URI is `sqlite:///mlflow.db` (created automatically). To explore runs:

```bash
mlflow ui
# Open http://localhost:5000
```

---

## Adding Your Own Ontology Pair

1. **Place your ontology files** in a new dataset folder:
   ```
   datasets/my_dataset/source_onto.owl
   datasets/my_dataset/target_onto.owl
   ```

2. **Add a reference alignment** (EDOAL or RDF format):
   ```
   references/my_dataset/source_onto-target_onto.owl
   ```

3. **Run the pipeline** pointing to your dataset:
   ```bash
   python -m stage.pipeline \
       dataset_name=my_dataset \
       ontology_source_name=source_onto \
       ontology_target_name=target_onto
   ```

The system accepts both `.owl` and `.rdf` file extensions and auto-detects them.

---

## Project Structure

```
OM_System/
├── pyproject.toml                  # project metadata and dependencies
├── download_benchmark.sh           # downloads datasets and references
├── datasets/                       # OWL/RDF ontology files
├── references/                     # ground-truth EDOAL alignments
├── samples/                        # few-shot prompt examples
├── outputs/                        # generated at runtime
└── stage/
    ├── config.yaml                 # main Hydra configuration
    ├── pipeline.py                 # OMPipeline class + Hydra entry point
    ├── preprocessing/              # ontology parsing and modularization
    │   ├── preprocessing.py
    │   ├── ontology_pair.py
    │   ├── ontology_parser.py
    │   ├── networkx_parser.py
    │   └── rdflib_to_networkx.py
    ├── modularization/             # PageRank / PPR / Sample modularizers
    │   ├── modularizer.py
    │   └── pagerank.py
    ├── verbalization/              # IRI → text conversion
    │   ├── build_verbalizer.py
    │   ├── base_verbalizer.py
    │   ├── label_verbalizer.py
    │   └── natural_verbalizer.py
    ├── candidate_generation/       # embedding + similarity search
    │   ├── candidate_generation.py
    │   ├── build_embedding_model.py
    │   ├── embedding_model_transformers.py
    │   ├── embedding_model_ollama.py
    │   └── embedding_model_llamacpp.py
    ├── matching/                   # LLM prompting
    │   ├── matching.py
    │   ├── build_llm_model.py
    │   ├── prompt_generator.py
    │   ├── prompt_generator_transformers.py
    │   ├── prompt_generator_ollama.py
    │   ├── llm_model_vllm.py
    │   ├── llm_model_ollama.py
    │   └── llm_model_llamacpp.py
    ├── postprocessing/             # XML merging and repair
    │   ├── postprocessing.py
    │   ├── answer_extractor.py
    │   └── xml_parser_refinement.py
    └── evaluation/                 # precision / recall / F1
        ├── evaluation.py
        ├── complex_evaluate.py
        ├── edoal_parser.py
        └── tree_similarity.py
```
