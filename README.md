# Temporal Generalization in Chinese Policy-Signal Classification

This repository contains the code and aggregate results for a frozen
evaluation of DeepSeek on the human-labeled portion of CAPC-CG.

## Research question

When DeepSeek receives recent human-labeled examples, does its five-signal
classification performance decline on earlier Chinese central-government
directives?

## Design

- Model: `deepseek-v4-pro`
- Official CAPC-CG two-stage workflow:
  - Level 1: affirmative directive (W), prohibition (R), or no directive (N)
  - Level 2 for W: Black (B), Yellow (Y), Charcoal (C), or Grey (G)
- Prompt: based on Appendix D of the CAPC-CG paper
- Few-shot examples: 17 recent human-gold training examples
- Development set: none
- Prompt/evaluation text overlap: 0
- Prompt/evaluation document overlap: 0
- Evaluation: all eligible, metadata-matched, conflict-free human-gold rows
- Evaluation population: 1,584 rows across 624 policy documents
- End-to-end test: 381 recent and 925 earlier directives
- Primary metric: label-standardized accuracy
- Uncertainty: 10,000 label-stratified document-cluster bootstrap repetitions

## Result

Recent label-standardized accuracy was 0.555. Earlier accuracy was 0.544.
The recent-minus-earlier difference was +0.010, with a 95% cluster-bootstrap
interval of [-0.053, +0.099]. The experiment therefore did not find reliable
evidence of lower performance on earlier policy language.

The stage decomposition was more informative. Level-1 directive screening
was better on recent text, while oracle-routed Level-2 signal classification
was better on earlier text. Temporal generalization was stage-specific.

## Repository map

- `src/prepare.py`: constructs the locked evaluation census and few-shot sets
- `src/lock.py`: records pre-run hashes
- `src/run.py`: executes the two-stage DeepSeek classification
- `src/audit.py`: checks routing, completeness, duplicates, hashes, and secrets
- `src/evaluate.py`: computes metrics and document-cluster bootstrap intervals
- `tests/`: unit tests for output parsing, routing, and metrics
- `config/protocol.json`: frozen confirmatory protocol
- `config/prompts.json`: prompt definitions
- `results/summary/formal_metrics.json`: aggregate formal results
- `report/main.tex`: complete research report

## Reproduction

Create a Python environment, install the dependencies, and add a local
`.env` file with the required API and dataset credentials. Then run:

```powershell
python src/prepare.py
python src/lock.py
python src/run.py
python src/audit.py
python src/evaluate.py
python -m unittest discover -s tests -v
```

The preparation step expects an authorized local copy of CAPC-CG in the path
configured by the script. Do not commit `.env` or generated row-level files.

## Data access and licensing

CAPC-CG is a gated dataset intended for non-commercial research. This
repository does **not** redistribute policy text, human-labeled prompt
examples, raw model responses, or row-level error files. Request access from
the [CAPC-CG dataset page](https://huggingface.co/datasets/Baron-Sun/CAPC-CG_V1.0)
and cite the [ACL 2026 paper](https://aclanthology.org/2026.acl-long.42/).

## Integrity

The protocol, prompt, code, and evaluation manifest were hashed before the
prediction file existed. The final run had zero API errors, invalid outputs,
missing required outputs, duplicate identifiers, routing mismatches, or
locked-file hash mismatches.

