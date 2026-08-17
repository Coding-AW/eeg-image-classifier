# Leakage-control specifications

Chance top-1 is 0.5%. Each stochastic or permuted control had to remain below the conservative 5%
ceiling, and their aggregate mean had to remain below 2%. Passing supports only the boundary tested.

| Control | Information deliberately broken | Implementation and leakage route tested | Expected result | What passing means—and does not mean |
| --- | --- | --- | --- | --- |
| Random ranking | All learned query–candidate information | Generate candidate order from the locked seed | About 0.5% top-1 | Metric/chance implementation behaves sensibly; it does not test fitting |
| Shuffled training pairs, three seeds | Correct EEG–image correspondence during fitting | Independently permute reference visual targets before Ridge fitting | About 0.5% | High scores require genuine reference pairing; it does not audit upstream construction |
| Shuffled query rows | Query-to-answer alignment | Permute the 200 averaged EEG queries while keeping the gallery fixed | About 0.5% | Scores are not supplied by sorted query position |
| Candidate-row permutation | Template-to-concept alignment | Permute template vectors independently of truth | About 0.5% | Scores require correct gallery mapping |
| Candidate-label permutation | Candidate meaning | Keep vectors but randomly relabel candidates | About 0.5% | Labels are not recoverable from candidate order |
| Stimulus-ID permutation | Trial/query identity | Break the immutable stimulus mapping independently | About 0.5% | Correct IDs are necessary; physical row order is insufficient |
| Category-name permutation | Human-readable concept mapping | Randomly remap names while preserving numeric arrays | About 0.5% | Names are not covertly providing answers |
| Row-order invariance | Physical storage position | Randomize raw test rows before ID grouping and compare averages/ranks exactly | No change | Grouping is ID-based; it does not validate the ID source itself |
| Gallery filename audit | Same file appearing in EEG and gallery | Compare immutable filenames against all EEG stimuli | Zero overlap | Exact filenames are excluded; renamed duplicates require the hash audit |
| Gallery SHA-256 audit | Renamed or copied EEG images | Compare full file-content hashes | Zero overlap | Exact content is excluded; semantically similar images remain intentionally allowed |
| Independent ranking | A scoring or rank-code error | Recompute all cosine scores/ranks with a separate implementation | Exact agreement | Production ranking is reproduced; both still depend on correct inputs |
| Bundle hash check | Evaluation modifying the fitted model | Hash every model before and after evaluation | No hash change | Evaluation did not mutate saved models; it does not prove fitting was clean |

Observed aggregate top-1 was 0.476%; the largest individual broken-information control was 2.5%.
All repository-controlled gates passed. Upstream construction of the distributed BraVL EEG arrays
cannot be reconstructed here and remains unresolved.
