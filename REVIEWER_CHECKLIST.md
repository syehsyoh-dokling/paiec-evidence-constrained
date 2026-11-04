# Reviewer checklist for PAIEC technical reports

Derived from the organizers' stated emphasis (technical soundness, clarity, reproducibility,
quality of experimental analysis; novelty secondary) and from the automated gate used in this
entry (`code/localsim/qa_gate.py`). Each item is answered yes / partly / no with one line of
evidence from the report or the released code. Released with this entry as a community resource.

## 1. Format and required content
- [ ] NeurIPS template; main text approximately 4–8 pages; references and appendices unlimited.
- [ ] Data, submitted approach, implementation, experimental setup, results and limitations are each described.
- [ ] Instructions sufficient to rerun the method (commands, inputs, expected outputs, runtime).
- [ ] Disclosure of every training source and of any LLM coding-assistant use.
- [ ] Supplementary material contains the code used to develop and run the submission.

## 2. Data and rules
- [ ] All training data come from the benchmark inventory / measurement-db; the channel and revision are stated.
- [ ] No hidden-pool data, no de-anonymisation of formative identifiers, no credentials in archives.
- [ ] Any data that differ from measurement-db are released with the supplement.

## 3. Evidence for every performance claim
- [ ] Is the comparison paired (same subject–item pairs for every variant), or are the numbers from different samples?
- [ ] Does any claim rest on a single hosted formative score? (Formative samples are redrawn per submission.)
- [ ] Are standard errors reported, and at which level (pairs, folds, benchmarks)? Do conclusions survive the coarser level?
- [ ] Was the selection of the submitted variant done on the same data it is evaluated on (nested selection or held-out confirmation)?
- [ ] With many candidates, is multiple testing addressed?
- [ ] Are the organizers' baselines run on the same pairs as the method?

## 4. Leakage
- [ ] For benchmark-level cold start, are fitted artifacts (priors, item regressors, centroids) refit without the held-out benchmark?
- [ ] Are acquisition and evaluation items disjoint per pair?
- [ ] Do context labels (same subject on other benchmarks) come from benchmarks that the priors were fitted on? If so, is that stated?

## 5. Measurement-science content
- [ ] Calibration reported (ECE, reliability diagrams) in addition to Brier.
- [ ] Brier decomposed (reliability / resolution / uncertainty) where a mechanism is claimed.
- [ ] Budget curve shown per budget, not only the ALC aggregate.
- [ ] Item-level claims separated from pair-level claims (constant-at-pass-rate oracle).

## 6. Ablation and controls
- [ ] Each component's contribution measured by removal on the same pairs.
- [ ] Acquisition policy compared with random acquisition.
- [ ] Any LLM/agent component compared with a non-LLM control on the same search space.

## 7. Honesty and limitations
- [ ] Negative results stated in the main text, not only in appendices.
- [ ] Tooling inconsistencies and failed candidates reported.
- [ ] Limitations name what was not compared (toolkits, literature models) and why.

## 8. Reproducibility in practice
- [ ] Paths and credentials externalised (environment variables), not hardcoded.
- [ ] Every number in the report traceable to a released script and result file.
- [ ] Submission archive passes the organizers' `check_submission_zip.py`.
