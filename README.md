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
src/
├── README.md                                    # This file
│
├── utils.py                                     # Shared utilities (LLM client, etc.)
│
├── persuasion-misaligned-coding-unified.ipynb  # Coding tasks (misaligned)
├── persuasion-misaligned-web-unified.ipynb     # Web research (misaligned)
├── persuasion-aligned-web-unified.ipynb        # Web research (aligned)
│
└── visualization-unified.ipynb                  # Comprehensive visualization & analysis
```

## Unified Notebooks

All experiments have been consolidated into **three main notebooks**, each supporting two experiment modes:

### 1. Misaligned Coding (`persuasion-misaligned-coding-unified.ipynb`)

**Task**: Python coding problems (KodCode dataset)
**Persuasion**: Unrelated opinions (e.g., social media liability, tenure reform)
**Hypothesis**: Opinion persuasion → behavioral changes in coding (errors, revisions, duration)

**Modes**:
- `opinion_persuasion`: 7-step pipeline with opinion tracking
- `prefill_only`: Direct prefill with belief state (P/NP/C0)

### 2. Misaligned Web (`persuasion-misaligned-web-unified.ipynb`)

**Task**: Web research (TREC Session Track tasks)
**Persuasion**: Unrelated opinions (same as coding)
**Hypothesis**: Opinion persuasion → behavioral changes in web surfing (URLs, domains, search patterns)

**Modes**:
- `opinion_persuasion`: 7-step pipeline with opinion tracking
  - **Conditions**: `baseline`, `neutral_injection`, persuasion tactics
- `prefill_only`: Direct prefill with belief state (P/NP/C0)

### 3. Aligned Web (`persuasion-aligned-web-unified.ipynb`)

**Task**: Web research (TREC Session Track tasks)
**Persuasion**: Task-aligned execution strategies (breadth vs depth)
**Hypothesis**: Execution preference persuasion → direct behavioral changes

**Behavior Policies**:
- **Breadth**: Few sources vs. diverse sources
- **Depth**: Skim quickly vs. careful extraction

**Modes**:
- `opinion_persuasion`: 7-step pipeline with preference tracking
- `prefill_only`: Direct prefill with execution preference (P/NP/C0)

### 4. Visualization (`visualization-unified.ipynb`)

Comprehensive analysis and visualization toolkit:
- Friction score calculation (percentile-based aggregation)
- Statistical tests (Mann-Whitney U, effect sizes)
- Core plots (friction, behavioral heatmaps, comparisons)
- Support for all experiment types and modes

## Experiment Modes

### Opinion-Persuasion Mode (7-Step Pipeline)

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

### Prefill-Only Mode

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

# AutoGen
autogen-agentchat>=0.4.0
autogen-ext[web-surfer]>=0.4.0

# Visualization
matplotlib
seaborn

# Data
datasets
```

### API Keys

Set environment variables:

```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GEMINI_API_KEY="your-key"
```

Or configure in notebook cells:

```python
os.environ["OPENAI_API_KEY"] = "your-key"
```

## Usage

### Quick Start

1. **Choose an experiment notebook**:
   - Coding tasks: `persuasion-misaligned-coding-unified.ipynb`
   - Web (misaligned): `persuasion-misaligned-web-unified.ipynb`
   - Web (aligned): `persuasion-aligned-web-unified.ipynb`

2. **Set experiment mode** in the configuration cell:
   ```python
   EXPERIMENT_MODE = "opinion_persuasion"  # or "prefill_only"
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
   - Open `visualization-unified.ipynb`
   - Load your data files
   - Run statistical tests and generate plots

### Example: Running a Coding Experiment

```python
# In persuasion-misaligned-coding-unified.ipynb

# 1. Set mode
EXPERIMENT_MODE = "opinion_persuasion"

# 2. Configure
personas = ["gpt", "claude"]
tactics = ["evidence_based", "logical_appeal", "baseline"]
N_RUNS = 5
N_CODE_PER_RUN = 10

# 3. Run
for run_i in range(N_RUNS):
    df = await run_batch_coding(
        run_id=f"run{run_i:02d}",
        personas=personas,
        tactics=tactics,
        model_client=model_client,
        writer_client=writer_client,
        ds=ds,
        n_code=N_CODE_PER_RUN,
        seed=42 + run_i,
        pairs=CLAIM_PAIRS,
        experiment_mode=EXPERIMENT_MODE,
    )

    df.to_json(f"results_run{run_i}.jsonl", orient="records", lines=True)
```

### Example: Analyzing Results

```python
# In visualization-unified.ipynb

# Load data
df = pd.read_json("results_run00.jsonl", lines=True)

# Compute friction score
df = compute_friction_score(df, CODING_RAW_METRICS)

# Filter to treatment conditions
df_nobase = filter_baseline(df)

# Statistical test
print(pooled_np_p_test(df_nobase))

# Visualizations
plot_friction_by_persuasion(df_nobase)
plot_friction_by_tactic(df_nobase)
plot_behavioral_heatmap(df_nobase, CODING_RAW_METRICS)
```

## Data Formats

### Output Schema (JSONL)

Each trial produces a row with:

#### Core Metadata
- `ts`: Timestamp
- `trial_id`: Unique trial identifier
- `persona`: Model persona/backbone
- `tactic`: Persuasion tactic used
- `experiment_mode`: "opinion_persuasion" or "prefill_only"

#### Opinion Measurements (opinion_persuasion mode)
- `prior_choice`: Initial opinion (A/B)
- `post_choice`: Opinion after persuasion
- `final_choice`: Opinion after distractors
- `persuaded`: 1 if post ≠ prior, 0 otherwise
- `persisted`: 1 if persuaded and final == post

#### Prefill (prefill_only mode)
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
- `gemini`: Empathetic, supportive, pragmatic
- `neutral`: Neutral, concise, practical

## Key Features

✅ **Unified Codebase**: Single notebook per experiment type
✅ **Mode Switching**: Easy toggle between opinion-persuasion and prefill-only
✅ **Persistent Agent**: Same agent throughout all 7 steps
✅ **Comprehensive Metrics**: 8+ behavioral metrics per task type
✅ **Statistical Rigor**: Mann-Whitney U tests, effect sizes, persona analysis
✅ **Publication-Ready Viz**: Clean plots and heatmaps
✅ **Reproducible**: Seeds, logging, trace files

## Citation

If you use this code in your research, please cite:

```bibtex
@article{jeong2025persuasion,
  title={Persuasion Persistence and Behavioral Misalignment in Large Language Models},
  author={Jeong, Hyejun and others},
  journal={TBD},
  year={2025}
}
```

## License

[Specify your license here]

## Contact

For questions or issues, please contact:
- Hyejun Jeong: hjeong@umass.edu
- [Add collaborators]

## Acknowledgments

- **KodCode Dataset**: Python coding problems
- **TREC Session Track**: Web research tasks
- **AutoGen Framework**: Multi-agent orchestration
- **Claude, GPT, Llama, etc.**: LLM providers

---

**Note**: This README describes the unified, refactored codebase. Legacy notebooks (visualization-v2.ipynb, comprehensive_vis.ipynb, etc.) have been consolidated and can be archived.
