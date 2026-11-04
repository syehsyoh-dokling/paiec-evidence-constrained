# Autopilot report: random_slot

Candidates: 16 | PASS 7 | HOLD 8 | REJECT 1 | confirmed on full protocol: 1 | elapsed 2494 s
Staged for upload: 7 in `quota_random_slot` | upload: all staged candidates submitted

| candidate | stage | verdict | dALC | SE | folds+ | reason |
|---|---|---|---|---|---|---|
| rnd_005_b0.zip | screen | PASS_UPLOAD | -0.0022 | 0.0002 | 5 |  |
| rnd_006_b0.zip | screen | PASS_UPLOAD | -0.0021 | 0.0002 | 5 |  |
| rnd_001_b0.zip | screen | PASS_UPLOAD | -0.0018 | 0.0002 | 5 |  |
| rnd_003_b0.zip | screen | PASS_UPLOAD | -0.0017 | 0.0001 | 5 |  |
| rnd_007_b0.zip | screen | PASS_UPLOAD | -0.0015 | 0.0001 | 5 |  |
| rnd_002_b0.zip | screen | PASS_UPLOAD | -0.0015 | 0.0001 | 5 |  |
| rnd_004_b0.zip | screen | PASS_UPLOAD | -0.0010 | 0.0001 | 5 |  |
| rnd_013_peer_shift.zip | full | HOLD | -0.0001 | 0.0002 | 4 |  |
| rnd_012_low_budget_s.zip | screen | HOLD | -0.0000 | 0.0000 | 2 |  |
| rnd_008_b0.zip | screen | REJECT |  |  |  | LOBO error: b0 eval failed on agentdojo: Warning: You are sending unau |
| rnd_010_low_budget_s.zip | screen | HOLD | +0.0000 | 0.0001 | 2 |  |
| rnd_011_low_budget_s.zip | screen | HOLD | +0.0000 | 0.0002 | 1 |  |
| rnd_009_low_budget_s.zip | screen | HOLD | +0.0000 | 0.0002 | 1 |  |
| rnd_014_peer_shift.zip | screen | HOLD | +0.0002 | 0.0003 | 2 |  |
| rnd_016_peer_shift.zip | screen | HOLD | +0.0004 | 0.0003 | 1 |  |
| rnd_015_peer_shift.zip | screen | HOLD | +0.0004 | 0.0002 | 1 |  |

## Rules learned for the student
- B0: 7 proposals passed; explore around them: null; null; null
- LOW_BUDGET_SHRINK: every proposal so far is indistinguishable from v153 (|dALC| < 0.0002); propose values near the ends of the ranges, not near the baseline

## Needs a human
- rnd_008_b0.zip: LOBO error: b0 eval failed on agentdojo: Warning: You are sending unauthenticated requests to the HF Hub. Please set a H