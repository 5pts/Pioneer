# CAPC-CG Temporal Generalization — Code

This repository contains only the Python code used for a two-stage temporal
generalization experiment on the gated CAPC-CG policy corpus.

No dataset text, prompt examples, API responses, row-level predictions,
research results, or report files are included.

## What the code does

The pipeline tests whether a fixed DeepSeek classifier behaves differently on
recent and earlier Chinese central-government policy directives.

1. `src/prepare.py` builds a leakage-free evaluation census from an authorized
   local CAPC-CG copy.
2. `src/audit.py` verifies class balance, time ranges, unique IDs, and zero
   prompt/evaluation text or document overlap.
3. `src/lock.py` freezes hashes before predictions exist.
4. `src/run.py` performs the official two-stage classification:
   - Level 1: affirmative directive (W), prohibition (R), or no directive (N)
   - Level 2 for W: Black (B), Yellow (Y), Charcoal (C), or Grey (G)
5. `src/evaluate.py` computes end-to-end and stage-specific metrics and runs a
   label-stratified document-cluster bootstrap.

## Repository contents

```text
src/
  prepare.py     Build the formal evaluation inputs
  audit.py       Validate inputs and leakage controls
  lock.py        Freeze pre-run file hashes
  run.py         Execute resumable DeepSeek predictions
  evaluate.py    Compute metrics and uncertainty
tests/
  test_run.py
  test_metrics.py
requirements.txt
```

## Private local inputs

The scripts expect the following files locally. They are intentionally excluded
from GitHub:

```text
.env
config/protocol.json
config/prompts.json
data/processed/
results/
../external_corpora/capc-cg-v1.0/
```

The `.env` file should contain:

```text
DEEPSEEK_API_KEY=your_key_here
```

Never commit the API key or any gated CAPC-CG content.

## Run order

Install Python 3.11 or newer and the dependencies:

```powershell
python -m pip install -r requirements.txt
```

After placing authorized inputs in the expected local paths:

```powershell
python src/prepare.py
python src/audit.py
python src/lock.py
python src/run.py
python src/audit.py
python src/evaluate.py
```

Run the unit tests with:

```powershell
python -m unittest discover -s tests -v
```

## Safety and reproducibility

- The prediction file is append-only, so interrupted API runs can resume.
- Invalid model outputs count as classification errors.
- Transient API failures use bounded retry logic.
- Prompt texts and their source documents are excluded from evaluation.
- Document-cluster bootstrap sampling respects within-document dependence.
- The execution lock prevents post-result changes from being presented as
  preregistered choices.

CAPC-CG is a gated dataset for non-commercial research. Request access from
the [dataset page](https://huggingface.co/datasets/Baron-Sun/CAPC-CG_V1.0)
and cite the [ACL 2026 paper](https://aclanthology.org/2026.acl-long.42/).

