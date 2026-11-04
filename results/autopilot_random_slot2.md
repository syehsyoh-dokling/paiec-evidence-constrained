# Autopilot report: random_slot2

Candidates: 32 | PASS 8 | HOLD 21 | REJECT 3 | confirmed on full protocol: 1 | elapsed 10146 s
Staged for upload: 8 in `quota_random_slot2` | upload: stopped at the daily quota

| candidate | stage | verdict | dALC | SE | folds+ | reason |
|---|---|---|---|---|---|---|
| rnd_001_b0.zip | screen | PASS_UPLOAD | -0.0021 | 0.0002 | 5 |  |
| rnd_007_b0.zip | screen | PASS_UPLOAD | -0.0020 | 0.0002 | 5 |  |
| rnd_006_b0.zip | screen | PASS_UPLOAD | -0.0020 | 0.0002 | 5 |  |
| rnd_003_b0.zip | screen | PASS_UPLOAD | -0.0018 | 0.0002 | 5 |  |
| rnd_002_b0.zip | screen | PASS_UPLOAD | -0.0016 | 0.0001 | 5 |  |
| rnd_008_b0.zip | screen | PASS_UPLOAD | -0.0014 | 0.0001 | 5 |  |
| rnd_004_b0.zip | screen | PASS_UPLOAD | -0.0011 | 0.0001 | 5 |  |
| rnd_005_b0.zip | screen | PASS_UPLOAD | -0.0011 | 0.0001 | 5 |  |
| rnd_012_low_budget_s.zip | screen | HOLD | -0.0000 | 0.0000 | 2 |  |
| rnd_018_low_budget_s.zip | screen | HOLD | -0.0000 | 0.0000 | 2 |  |
| rnd_010_low_budget_s.zip | screen | HOLD | +0.0000 | 0.0001 | 2 |  |
| rnd_017_low_budget_s.zip | screen | HOLD | +0.0000 | 0.0000 | 2 |  |
| rnd_014_low_budget_s.zip | screen | HOLD | +0.0000 | 0.0000 | 2 |  |
| rnd_013_low_budget_s.zip | screen | HOLD | +0.0000 | 0.0001 | 1 |  |
| rnd_016_low_budget_s.zip | screen | HOLD | +0.0000 | 0.0002 | 1 |  |
| rnd_011_low_budget_s.zip | screen | HOLD | +0.0000 | 0.0002 | 1 |  |
| rnd_032_peer_shift.zip | full | HOLD | +0.0000 | 0.0002 | 4 |  |
| rnd_020_low_budget_s.zip | screen | HOLD | +0.0000 | 0.0002 | 1 |  |
| rnd_015_low_budget_s.zip | screen | HOLD | +0.0000 | 0.0002 | 1 |  |
| rnd_009_low_budget_s.zip | screen | HOLD | +0.0000 | 0.0002 | 1 |  |
| rnd_019_low_budget_s.zip | screen | HOLD | +0.0000 | 0.0002 | 1 |  |
| rnd_026_peer_shift.zip | screen | HOLD | +0.0001 | 0.0002 | 2 |  |
| rnd_031_peer_shift.zip | screen | HOLD | +0.0001 | 0.0003 | 2 |  |
| rnd_028_peer_shift.zip | screen | HOLD | +0.0002 | 0.0002 | 1 |  |
| rnd_029_peer_shift.zip | screen | HOLD | +0.0002 | 0.0002 | 2 |  |
| rnd_024_peer_shift.zip | screen | HOLD | +0.0002 | 0.0002 | 1 |  |
| rnd_021_peer_shift.zip | screen | REJECT | +0.0002 | 0.0001 | 0 |  |
| rnd_022_peer_shift.zip | screen | HOLD | +0.0003 | 0.0002 | 1 |  |
| rnd_030_peer_shift.zip | screen | HOLD | +0.0003 | 0.0003 | 2 |  |
| rnd_025_peer_shift.zip | screen | REJECT | +0.0003 | 0.0001 | 0 |  |
| rnd_027_peer_shift.zip | screen | HOLD | +0.0004 | 0.0002 | 0 |  |
| rnd_023_peer_shift.zip | screen | REJECT | +0.0004 | 0.0001 | 1 |  |

## Rules learned for the student
- B0: 8 proposals passed; explore around them: null; null; null
- LOW_BUDGET_SHRINK: every proposal so far is indistinguishable from v153 (|dALC| < 0.0002); propose values near the ends of the ranges, not near the baseline

## Needs a human
- nothing