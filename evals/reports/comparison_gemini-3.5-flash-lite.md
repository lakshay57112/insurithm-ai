# Comparison: agent vs baseline

Model: `gemini-3.5-flash-lite`. Run: 2026-10-26 18:20 UTC. Cases: 15 (10 appeal, 5 no-appeal).

| Metric | agent | baseline |
| --- | --- | --- |
| Pool recall (expected passages retrieved) | 0.938 | 0.625 |
| Citation correctness (valid / proposed) | 1.000 | 1.000 |
| Unsupported-claim rate, as proposed by the model | 0.059 | 0.000 |
| Unsupported-claim rate, as shipped to the user | 0.000 | 0.000 |
| Answer correctness, appeal cases | 0.600 | 0.300 |
| Evidence coverage, appeal cases | 0.688 | 0.375 |
| Abstained correctly, no-appeal cases | 1.000 | 1.000 |
| False abstention, appeal cases | 0.000 | 0.200 |
| Model calls per case | 4.670 | 1.000 |
| Cases that failed with an error | 0 | 0 |

The policy and the cases are synthetic and were written by the same person, the set is small, and one model on one run is one sample. Read these as a check that the system behaves as designed, not as a general accuracy figure.
