# Metric dictionary

Every metric below is descriptive unless the report explicitly identifies it as a primary outcome.

| Metric | What it is | Why it was measured | How to read it | Limitation |
| --- | --- | --- | --- | --- |
| Rows | Number of EEG or image-feature examples | Confirms expected data were loaded | Must match the declared split | Correct count does not prove correct identity |
| Classes | Number of distinct concepts | Checks task size and balance | Expected: 1,654 train and 200 test | Labels can still be misaligned |
| Width | Values in each vector | Confirms representation shape | EEG must be 561; visual width depends on encoder | Says nothing about useful information |
| Finite | Whether all values are real and bounded | Detects NaN/inf corruption | Must be true | Cannot detect plausible but wrong values |
| Mean | Average feature value | Reveals large offsets between splits/participants | Similar scale reduces avoidable shift | Near-zero mean does not imply clean data |
| Standard deviation | Typical spread around the mean | Checks scale and whether averaging reduces variability | Lower averaged spread is consistent with noise reduction | Lower is not always better |
| Effective rank | Approximate number of substantially used directions | Measures practical dimensionality rather than raw width | Higher means information is spread over more directions | Does not identify which directions aid decoding |
| Leading-component energy | Fraction of variation in the strongest direction | Detects domination by one shared pattern | High values suggest a common direction may dominate | The direction may be meaningful or nuisance |
| Within-stimulus variance | Variation among repeated EEG recordings of one stimulus | Quantifies repeat noise available for averaging to reduce | Lower means repetitions agree more | Mixes neural and non-neural variation |
| Similarity to 80-repeat mean | Cosine agreement between a subset average and the full average | Shows how quickly a stable estimate forms | Values approaching 1 indicate convergence | The full mean is not independent ground truth |
| First/last-40 agreement | Cosine similarity between two disjoint half averages | Tests repeatability with independent trial sets | Higher means stable response structure | Halves share participant and acquisition setting |
| Total variance | Sum of variance over feature dimensions | Checks whether a space is collapsed or scale-dominated | Compare only within consistently normalized spaces | Does not measure class usefulness directly |
| Feature norm mean/SD | Typical vector length and its variability | Confirms normalization and catches unusual scale | Unit-normalized vectors should have norm near 1 | Direction can still be wrong |
| Common-direction energy | Strength of the mean direction shared by examples | Detects broad structure common to many images | High values can reduce discriminative angular differences | Shared structure may still be useful |
| Within-class cosine | Average directional similarity of images from one concept | Measures concept compactness | Higher means same-concept images cluster | Can rise from a global common direction |
| Between-class cosine | Average similarity between different concepts | Measures crowding | Lower values mean concepts are more distinct | Average can hide difficult local neighbours |
| Standardized separation | Within-class minus between-class similarity, scaled by variation | Compares class signal with background spread | Higher means cleaner class separation | Not a decoder accuracy estimate |
| Nearest-prototype accuracy | Fraction of reference images closest to their own concept mean | Tests whether geometry supports simple retrieval | Higher means prototypes represent classes well | Uses reference data and is descriptive only |
| Prototype margin | Correct-template similarity minus strongest wrong-template similarity | Measures decision safety | Positive/larger margins mean less ambiguity | Mean margins can hide hard classes |
| Hubness skew | Unevenness in how often templates become nearest neighbours | Detects “hub” candidates attracting many predictions | Larger positive skew means more hubness | Does not identify the cause |
| Top-1 | Fraction with correct concept ranked first | Primary strict retrieval result | Chance is 1/200 = 0.5% | Ignores all ranks below first |
| Top-5 | Fraction with correct concept in first five | Primary broader retrieval result | Chance is 5/200 = 2.5% | Does not distinguish ranks 1–5 |
| 95% confidence interval | Range from participant mean, sample SD, and Student-t critical value | Shows uncertainty across people | Narrower means participant means are more consistent | With ten people, intervals remain approximate |
| Exact sign-flip p-value | Probability of participant differences at least this extreme under symmetric zero effect | Tests paired model differences without treating trials as people | Smaller values oppose the zero-difference model | Not effect size or practical importance |
| Holm-adjusted p-value | Step-down correction across the two primary comparisons | Limits family-wise false positives | Compare with the chosen alpha after correction | Does not correct unplanned analyses |
