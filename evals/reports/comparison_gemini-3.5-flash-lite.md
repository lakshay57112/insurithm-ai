# Comparison: agent vs baseline

Model: `gemini-3.5-flash-lite`. Run: 2026-10-26 18:46 UTC. Cases: 15 (10 appeal, 5 no-appeal).

| Metric | agent | baseline |
| --- | --- | --- |
| Pool recall (expected passages retrieved) | 0.938 | 0.625 |
| Citation correctness (valid / proposed) | 1.000 | 1.000 |
| Unsupported-claim rate, as proposed by the model | 0.067 | 0.000 |
| Unsupported-claim rate, as shipped to the user | 0.000 | 0.000 |
| Answer correctness, appeal cases | 0.400 | 0.300 |
| Evidence coverage, appeal cases | 0.562 | 0.375 |
| Abstained correctly, no-appeal cases | 0.800 | 1.000 |
| False abstention, appeal cases | 0.300 | 0.200 |
| Model calls per case | 4.530 | 1.000 |
| Cases that failed with an error | 0 | 0 |

Cases that were not fully right:

- agent C01: gave no appeal points; never retrieved 4.1 (page 4); retrieved but not cited 4.2 (page 4)
- agent C02: retrieved but not cited 3.2 (page 3)
- agent C03: retrieved but not cited 6.2 (page 6)
- agent C05: gave no appeal points; retrieved but not cited 6.7 (page 6)
- agent C07: retrieved but not cited letter (page 8)
- agent C08: gave no appeal points; retrieved but not cited 4.3 (page 4)
- agent A03: gave 2 appeal point(s) where the evidence is not enough
- baseline C01: retrieved but not cited 4.1 (page 4), 4.2 (page 4)
- baseline C02: retrieved but not cited 3.2 (page 3)
- baseline C03: never retrieved 6.4 (page 6); retrieved but not cited 6.2 (page 6)
- baseline C05: gave no appeal points; never retrieved 6.7 (page 6)
- baseline C07: gave no appeal points; never retrieved 2.4 (page 2), letter (page 8)
- baseline C09: never retrieved 3.5 (page 3)
- baseline C10: never retrieved 7.2 (page 7)

The policy and the cases are synthetic and were written by the same person, the set is small, and one model on one run is one sample. Read these as a check that the system behaves as designed, not as a general accuracy figure.
