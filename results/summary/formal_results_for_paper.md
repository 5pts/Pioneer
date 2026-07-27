# Formal experiment: paper-ready results

## Design

The final experiment used the official two-stage CAPC-CG annotation structure.
DeepSeek first classified each paragraph as affirmative (W), prohibitive (R),
or non-directive (N). All human-gold W paragraphs were also sent to the
four-class Level-2 task: Black (B), Yellow (Y), Charcoal (C), or Grey (G).
Running Level 2 for every gold-W paragraph made it possible to separate
Level-1 routing errors from true color-classification errors.

The system prompt was based on Appendix D of the CAPC-CG paper. The prompt
included 17 randomly selected recent examples from the human-gold training
data: three examples for each Level-1 label and two examples for each Level-2
label. No development set was used. Prompt examples and evaluation rows had
no shared text and no shared source documents.

Rather than selecting a small test sample, the experiment evaluated every
eligible, metadata-matched, conflict-free human-gold paragraph after excluding
the prompt documents. The resulting census contained 1,584 rows from 624
policy documents. The end-to-end five-class analysis contained 381 recent
rows and 925 earlier rows.

The primary outcome was label-standardized accuracy, calculated as the mean
recall across B, Y, C, G, and R. This metric prevents differences in label
frequency from determining the time comparison. Confidence intervals used
10,000 label-stratified document-cluster bootstrap repetitions.

## Main result

End-to-end label-standardized accuracy was 0.555 on recent directives and
0.544 on earlier directives. The recent-minus-earlier difference was +0.010,
or +1.0 percentage point. The 95% cluster-bootstrap confidence interval was
[-0.053, +0.099], and the bootstrap two-sided sign p-value was 0.698.

Overall accuracy was 0.541 for recent directives and 0.532 for earlier
directives. The difference was +0.009, with a 95% interval of
[-0.061, +0.079].

Macro-F1 was 0.567 for recent directives and 0.599 for earlier directives.
The difference was -0.032, with a 95% interval of [-0.095, +0.028].

All three intervals included zero. The formal experiment therefore did not
support the hypothesis that DeepSeek's performance falls on earlier policy
language.

## Stage decomposition

Level-1 performance was higher on recent text. Label-standardized Level-1
accuracy was 0.937 for recent task1 paragraphs and 0.799 for earlier task1
paragraphs.

The pattern reversed at Level 2. When every gold-W paragraph was sent directly
to the color classifier, label-standardized accuracy was 0.606 for recent text
and 0.648 for earlier text. Thus, recent language was easier for directive
screening, while earlier language was slightly easier for fine-grained color
classification.

These opposing stage-level patterns explain why the end-to-end temporal gap
was close to zero.

## Interpretation

The result is a valid null finding. It does not prove that policy language
never changes. It shows that, under this model, prompt, corpus, and time split,
the available human-gold evidence does not reveal a reliable decline on older
directives.

The earlier 45-pair pilot produced a larger positive gap. The full formal
census shows that estimate was not stable. The formal result should replace
the pilot result in the paper.

## Limits that must remain in the paper

- The matched recent gold data contain only 17 B paragraphs and 19 R
  paragraphs after prompt-document exclusion.
- The recent matched gold period ends in 2022, although the corpus extends to
  2023.
- The experiment uses one model and one fixed prompt.
- Model pretraining data are unknown, so the model cannot be described as
  learning only from the recent few-shot examples.
- A null temporal gap does not establish that policy language is historically
  unchanged. It only describes model performance on this classification task.
- Some gold texts match more than one source document within the same period.
  These cases do not change the time label, but exact document metadata can be
  uncertain.

## Reproducibility files

- Frozen protocol: `config/protocol.json`
- Frozen prompt: `config/prompts.json`
- Pre-run code and data hashes: `config/execution_lock.json`
- Evaluation census: `data/processed/evaluation_census.csv`
- Raw predictions: `results/raw/formal_predictions.jsonl`
- Complete metrics: `results/summary/formal_metrics.json`
- Error cases: `results/summary/formal_errors.csv`
- Result hashes: `results/summary/result_manifest.json`

