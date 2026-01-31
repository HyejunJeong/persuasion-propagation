# Persuasion Persistence & Behavioral Misalignment Study

## Overview

This repository contains the implementation of experiments investigating **persuasion persistence** and **behavioral misalignment** in large language models (LLMs). The study examines whether persuading an agent on one topic (or execution strategy) influences its behavior on subsequent, unrelated tasks.

### Research Questions

1. **Persuasion Persistence**: Do LLMs maintain persuaded beliefs after distractor tasks?
2. **Behavioral Misalignment**: Does opinion persuasion on topic X alter task behavior on unrelated topic Y?
3. **Aligned vs Misaligned**: How does persuading about the actual task (aligned) differ from persuading about unrelated topics (misaligned)?
4. **Persona Vulnerability**: Which personas/models are most vulnerable to which persuasion tactics?

## Repository Structure

```
persuasion_propagation/
├── README.md                     # This file
│
├── utils.py                      # Shared utilities (LLM client, constants, helpers)
├── vis.py                        # Visualization & analysis functions
│
├── notebook/                     # Jupyter notebooks
│   ├── opinion_change.ipynb                        # Opinion persistence experiments
│   ├── persuasion-misaligned-coding-unified.ipynb  # Coding tasks (misaligned)
│   ├── persuasion-misaligned-web-unified.ipynb     # Web research (misaligned)
│   ├── persuasion-aligned-web-unified.ipynb        # Web research (aligned)
│   └── visualization-unified.ipynb                 # Comprehensive visualization
│
├── results/                      # Experimental results (see Data section)
└── traces/                       # Execution traces (see Data section)
```

## Core Modules

### `utils.py` - Shared Utilities

Contains reusable functions and constants:

**Constants**:
- `PERSONAS`: 7 personality definitions (gpt, claude, llama, mistral, qwen, gemma, neutral)
- `TACTICS`: 5 persuasion tactics (logical_appeal, authority_endorsement, evidence_based, priming_urgency, anchoring)
- `RECALL_PROBE`: Memory probe text
- `CHOICE_RE`: Regex pattern for choice extraction

**Functions**:
- `LLMClient`: Universal LLM client supporting OpenAI, Anthropic, Google, Together, and HuggingFace
- `parse_choice()`: Extract A/B choices from text
- `generate_topic_persuasion_line_with_writer()`: Generate persuasive text using specified tactics
- `normalize_persuasion_df()`: Normalize different data schemas
- `aggregate_backbone()`: Aggregate persuasion data by tactic

### `vis.py` - Visualization & Analysis

Contains plotting and statistical analysis functions:

**Data Loading**:
- `load_jsonl()`: Load JSONL files with backbone labels
- `load_multiple_files()`: Concatenate multiple experiment files
- `filter_baseline()`: Remove baseline conditions

**Metrics**:
- `compute_normalized_metrics()`: Baseline-normalized metrics

**Statistical Tests**:
- `pooled_np_p_test()`: Mann-Whitney U test (not-persuaded vs persuaded)
- `persona_delta_summary()`: Per-persona statistical summary
- `tactic_summary()`: Per-tactic persuasion rates and effects

**Plotting**:
- `plot_coding_delta_boxplot()`: Box plots for coding metrics
- `plot_pct_box_grid()`: Grid of box plots by persona × tactic
- `plot_coding_delta_side_by_side()`: Side-by-side heatmaps (NP vs P)
- `plot_pc_difference()`: Persuasion-induced shift heatmaps

## Notebooks

### 1. Opinion Change (`opinion_change.ipynb`)

**Focus**: Single-agent opinion flip experiments with persistence tracking

**Pipeline**:
1. Prior opinion evaluation (A or B)
2. Generate persuasion targeting opposite stance
3. Persuasion + commitment loop (3 turns)
4. Post opinion evaluation → `persuaded = (post ≠ prior)`
5. Distractor questions (configurable: 1-8)
6. Final opinion evaluation → `persisted = (final == post)`
7. Recall probe

**Key Metrics**:
- `persuaded`: Opinion changed after persuasion
- `persisted`: Persuaded opinion maintained after distractors
- `prior_choice`, `post_choice`, `final_choice`: Opinion trajectory

### 2. Persuasion-Aligned Web (`persuasion-aligned-web-unified.ipynb`)

**Task**: Web research (TREC Session Track)
**Persuasion**: Task-aligned execution strategies (breadth vs depth)
**Hypothesis**: Execution preference persuasion → direct behavioral changes

**Behavior Policies**:
- **Breadth**: Few sources vs. diverse sources
- **Depth**: Skim quickly vs. careful extraction

**Modes**:
- `onthefly`: 7-step pipeline with preference tracking
- `prefill`: Direct prefill with execution preference (P/NP/C0)

### 3. Persuasion-Misaligned Coding (`persuasion-misaligned-coding-unified.ipynb`)

**Task**: Python coding problems (KodCode dataset)
**Persuasion**: Unrelated opinions (e.g., social media liability, tenure reform)
**Hypothesis**: Opinion persuasion → behavioral changes in coding

**Behavioral Metrics**:
- **TRS (Task Revision Score)**: Primary metric for coding task behavior
- **EVS (Exploration Variability Score)**: Measures variation in coding strategies
- Additional metrics:
  - `num_errors`: Code execution errors
  - `num_code_revisions`: Number of revisions made
  - `coding_duration_s`: Time spent coding
  - `revision_entropy`: Diversity of revision types
  - `strategy_switch_rate`: Strategy changes during solving
  - `overcommitment`: Persisting with failing approaches
  - `mean_revision_size`: Average revision magnitude
  - `final_revision_delta`: Size of final fix

**Modes**:
- `onthefly`: 7-step pipeline with opinion tracking
- `prefill`: Direct prefill with belief state (P/NP/C0)

### 4. Persuasion-Misaligned Web (`persuasion-misaligned-web-unified.ipynb`)

**Task**: Web research (TREC Session Track)
**Persuasion**: Unrelated opinions (same as coding)
**Hypothesis**: Opinion persuasion → behavioral changes in web surfing

**Behavioral Metrics**:
- `num_urls`: Total URLs visited
- `num_unique_urls`: Unique URLs visited
- `num_domains`: Number of distinct domains
- `domain_entropy`: Shannon entropy of domain distribution
- `num_searches`: Number of search queries
- `num_summaries`: Number of summaries generated
- `avg_latency_s`: Average action latency
- `total_duration_s`: Total task duration

**Modes**:
- `onthefly`: 7-step pipeline with opinion tracking
  - **Conditions**: `baseline`, `neutral_injection`, persuasion tactics
- `prefill`: Direct prefill with belief state (P/NP/C0)

### 5. Visualization (`visualization-unified.ipynb`)

Comprehensive analysis and visualization toolkit:
- Statistical tests (Mann-Whitney U, effect sizes)
- Core plots (behavioral heatmaps, comparisons)
- Support for all experiment types and modes

## Experiment Modes

### On-the-Fly Mode (7-Step Pipeline)

```
1. Prior Opinion    → Measure initial stance (A or B)
2. Persuasion       → Inject persuasive prompt
3. Commitment       → Reinforce new stance (3 turns)
4. Post Opinion     → Remeasure → persuaded = (post ≠ prior)
5. Distractors      → Unrelated questions
6. Final Opinion    → Remeasure → persisted = (final == post)
7. Task Execution   → Capture behavioral metrics
```

**Key Metrics**:
- `persuaded`: 1 if opinion changed after persuasion
- `persisted`: 1 if persuaded opinion maintained after distractors
- Behavioral metrics: task-specific (coding or web)

### Prefill Mode

```
1. Prefill Reminder → Prime with belief/preference state
2. Task Execution   → Capture behavioral metrics
```

**Conditions**:
- `P` (Persuaded): "You WERE persuaded to adopt..."
- `NP` (Not Persuaded): "You were exposed but NOT persuaded..."
- `C0` (Neutral): "You have NOT formed a preference..."

## Installation

### Requirements

```bash
# Python 3.11+
pip install -r requirements.txt
```

### Key Dependencies

```
# Core
pandas
numpy
scipy

# LLM APIs
openai
anthropic
google-generativeai
together

# AutoGen
autogen-agentchat>=0.4.0
autogen-ext[web-surfer]>=0.4.0

# Visualization
matplotlib
seaborn

# Data
datasets
huggingface_hub
```

### API Keys

Set environment variables:

```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GEMINI_API_KEY="your-key"
export TOGETHER_API_KEY="your-key"
```

Or configure in notebook cells:

```python
import os
os.environ["OPENAI_API_KEY"] = "your-key"
```

## Usage

### Quick Start

1. **Choose an experiment notebook** from `notebook/`:
   - Opinion persistence: `opinion_change.ipynb`
   - Coding tasks: `persuasion-misaligned-coding-unified.ipynb`
   - Web (misaligned): `persuasion-misaligned-web-unified.ipynb`
   - Web (aligned): `persuasion-aligned-web-unified.ipynb`

2. **Set experiment mode** in the configuration cell:
   ```python
   EXPERIMENT_MODE = "onthefly"  # On-the-Fly mode
   # or
   EXPERIMENT_MODE = "prefill"   # Prefill mode
   ```

3. **Configure models**:
   ```python
   ASSISTANT_MODEL = "gpt-4.1-nano"
   SURFER_MODEL = "gpt-4o-2024-08-06"
   WRITER_MODEL_ID = "openai:gpt-4.1-nano"
   ```

4. **Run experiments**:
   ```python
   df = await run_batch(
       personas=["gpt", "claude"],
       tactics=["evidence_based", "logical_appeal"],
       tasks=TASKS,
       experiment_mode=EXPERIMENT_MODE,
       n_per_cell=10,
       seed=42,
   )
   ```

5. **Analyze results**:
   - Open `notebook/visualization-unified.ipynb`
   - Load your data files
   - Run statistical tests and generate plots

### Example: Running an Opinion Change Experiment

```python
# In notebook/opinion_change.ipynb

from utils import LLMClient

# Configure
personas = ["gpt", "claude", "mistral"]
tactics = ["logical_appeal", "authority_endorsement", "evidence_based",
           "priming_urgency", "anchoring", "none"]
n_distractors = 8  # Number of distractor questions

# Initialize writer client for persuasion generation
WRITER_MODEL_ID = "openai:gpt-4.1-nano"
writer_client = LLMClient(WRITER_MODEL_ID)

# Run experiment
df = await run_batch(
    personas=personas,
    tactics=tactics,
    mode="no_reset",
    n_per_cell=1,  # Trials per (persona, tactic, claim_pair)
    n_distractors=n_distractors,
    out_csv=Path(f"results/opinion_d{n_distractors}_persist.csv"),
    seed=42,
    writer_client=writer_client,
    pairs=range(1, 29),  # All 28 claim pairs
)
```

### Example: Analyzing Results

```python
# In notebook/visualization-unified.ipynb

import pandas as pd
from vis import (
    filter_baseline,
    pooled_np_p_test,
    CODING_RAW_METRICS, WEB_RAW_METRICS
)

# Load data
df = pd.read_json("results/coding_results.jsonl", lines=True)

# Filter to treatment conditions
df_nobase = filter_baseline(df)

# Statistical test for TRS (Task Revision Score)
print(pooled_np_p_test(df_nobase, score_col="trs"))

# Statistical test for EVS (Exploration Variability Score)
print(pooled_np_p_test(df_nobase, score_col="evs"))
```

## Data & Results

### Experimental Data (Large Files)

Due to file size limitations, trace files and result files are hosted externally:

**📦 Google Drive**: [Persuasion Propagation Data](https://drive.google.com/drive/folders/1mrFY_EGa0KYDvEn2xfEUgkBC5jnVdBAS?usp=sharing)

This includes:
- `results/`: Experimental results (JSONL format)
- `traces/`: Full execution traces and logs
- Pre-computed analysis outputs

### Output Schema (JSONL)

Each trial produces a row with:

#### Core Metadata
- `ts`: Timestamp
- `trial_id`: Unique trial identifier
- `persona`: Model persona/backbone
- `tactic`: Persuasion tactic used
- `experiment_mode`: `"onthefly"` (On-the-Fly mode) or `"prefill"` (Prefill mode)

#### Opinion Measurements (On-the-Fly mode)
- `prior_choice`: Initial opinion (A/B)
- `post_choice`: Opinion after persuasion
- `final_choice`: Opinion after distractors
- `persuaded`: 1 if post ≠ prior, 0 otherwise
- `persisted`: 1 if persuaded and final == post

#### Prefill (Prefill mode)
- `prefill_condition`: P, NP, or C0
- `prefill_reminder`: Exact prefill text used

#### Task-Specific Metrics

**Coding**:
- `num_errors`, `num_code_revisions`, `coding_duration_s`
- `revision_entropy`, `strategy_switch_rate`, `overcommitment`
- `solution_strategy`, `protocol_violation`

**Web**:
- `num_urls`, `num_unique_urls`, `num_domains`
- `domain_entropy`, `num_searches`, `num_summaries`
- `avg_latency_s`, `total_duration_s`

## Persuasion Tactics

### Implemented Tactics

1. **Logical Appeal**: Reasoning and cause-effect logic
2. **Authority Endorsement**: Credible standards and expert practices
3. **Evidence-Based**: Empirical data and performance evidence
4. **Priming/Urgency**: Time pressure and urgency cues
5. **Anchoring**: Demanding goal first, then achievable version

### Control Conditions

- **Baseline**: No persuasion or injection
- **Neutral Injection**: Turn-matched neutral conversation (control for interaction)

## Personas

Defined personality prompts for different LLM styles:

- `gpt`: Cooperative, balanced, pragmatic
- `claude`: Thoughtful, articulate, helpful
- `llama`: Straightforward, efficient, task-focused
- `mistral`: Lively, curious, results-oriented
- `qwen`: Polite, structured, logical
- `gemma`: Empathetic, supportive, pragmatic
- `neutral`: Neutral, concise, practical

## Key Features

✅ **Unified Codebase**: Single notebook per experiment type  
✅ **Mode Switching**: Easy toggle between on-the-fly and prefill modes  
✅ **Persistent Agent**: Same agent throughout all 7 steps  
✅ **Comprehensive Metrics**: 8+ behavioral metrics per task type  
✅ **Statistical Rigor**: Mann-Whitney U tests, effect sizes, persona analysis  
✅ **Publication-Ready Viz**: Clean plots and heatmaps  
✅ **Reproducible**: Seeds, logging, trace files  
✅ **Modular Design**: Reusable utilities in `utils.py` and `vis.py`  

## Citation

If you use this code in your research, please cite:

```bibtex
@article{jeong2025persuasion,
  title={Persuasion Propagation in LLM Agents},
  author={Jeong, Hyejun, Houmansadr, Amir, Zilberstein, Shlomo, and Bagdasarian, Eugene},
  journal={TBD},
  year={2026}
}
```

## License

[Specify your license here]

## Contact

For questions or issues, please contact:
- Hyejun Jeong: hjeong@umass.edu

## Acknowledgments

- **KodCode Dataset**: Python coding problems
- **TREC Session Track**: Web research tasks
- **AutoGen Framework**: Multi-agent orchestration
- **Claude, GPT, Llama, etc.**: LLM providers

---

**Last Updated**: January 2026
