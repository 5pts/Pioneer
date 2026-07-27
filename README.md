# Temporal Generalization in CAPC-CG

Code for testing whether a fixed DeepSeek classifier performs differently on
recent and earlier Chinese central-government policy directives.

The experiment follows the two-stage CAPC-CG annotation scheme:

1. classify each paragraph as affirmative (W), prohibitive (R), or
   non-directive (N);
2. classify W paragraphs as Black (B), Yellow (Y), Charcoal (C), or Grey (G).

## Structure

```text
config/
  protocol.json  Frozen design and analysis choices
  prompts.json   Classification instructions
src/
  prepare.py     Build the evaluation census
  audit.py       Check labels, dates, IDs, and leakage
  lock.py        Record pre-run hashes
  run.py         Run resumable model inference
  evaluate.py    Compute metrics and uncertainty
tests/
  test_metrics.py
  test_run.py
```

## Reproduce

Requires Python 3.11 or newer and an authorized local copy of CAPC-CG.

```powershell
python -m pip install -r requirements.txt
python src/prepare.py
python src/audit.py
python src/lock.py
python src/run.py
python src/audit.py
python src/evaluate.py
```

Set `DEEPSEEK_API_KEY` in the environment or in a local `.env` file. Run tests
with:

```powershell
python -m unittest discover -s tests -v
```

The gated corpus, API responses, row-level predictions, and generated reports
are not committed. Aggregate results are reported separately.

CAPC-CG is available for non-commercial research from its
[dataset page](https://huggingface.co/datasets/Baron-Sun/CAPC-CG_V1.0).

