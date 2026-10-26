# Evaluation

There are two evaluations. The first measures retrieval alone. The second compares the agent with a non-agentic baseline on synthetic appeal cases.

## 1. Retrieval on policy questions

`data/gold/gold_questions.json` holds 36 questions about the synthetic policy.

- 30 are answerable. Each names one expected page, the expected clause and a short verbatim passage.
- 6 are unanswerable (the policy says nothing about them). They are not scored for retrieval.

A retrieved chunk is a **hit** only if it is on the expected page and contains the expected passage.

- **Recall@k:** share of answerable questions with a hit in the top k results.
- **MRR:** mean of 1 / (rank of the first hit); 0 if there is no hit in the top k.

```
python -m insurithm.evals.retrieval_eval --ks 1 3 5
```

BM25 keyword retrieval, 30 answerable questions, 54-chunk policy (per-question results in `evals/reports/retrieval_bm25.json`):

| Metric | Score |
| --- | --- |
| Recall@1 | 0.667 |
| Recall@3 | 0.933 |
| Recall@5 | 1.000 |
| MRR | 0.797 |

`tests/test_retrieval.py` holds a floor just below these numbers, so a change that makes retrieval worse fails the build.

## 2. Agent vs. baseline on appeal cases

### The cases

`data/gold/gold_cases.json` has 15 synthetic rejection or settlement letters for the fictional policy.

- 10 are **appeal** cases. The policy gives the policyholder something to cite (a deduction applied to a room inside the limit, a late-intimation rejection that clause 6.4 lets the insurer condone, a pending claim past the 30-day deadline, and so on). Each lists the 1 or 2 passages a good appeal should cite.
- 5 are **no-appeal** cases. In some the policy supports the insurer (treatment abroad, an ambulance paid at the stated limit, elective cosmetic surgery); in others the policy says nothing relevant (a premium request, a health check-up benefit). A trustworthy system should not produce an appeal point for these.

`tests/test_gold_cases.py` checks that every expected passage sits inside a single chunk on the stated page, that the letters parse, and that every clause a letter cites exists.

### The two systems

Both get the same letter and the same policy and are told to produce findings in the same format with the same rules.

- **Baseline:** search the policy once with the letter text, take the top 5 chunks, make one model call, ship everything the model wrote.
- **Agent:** the full workflow up to the human review step (it fetches the clauses the letter cites, the model chooses up to 4 further searches, drafts, and the citation checker filters). Only findings that pass the checker are shipped; if none pass, the result is `insufficient_evidence`.

### Grading

Every output is graded by the same function, `insurithm.verify.verify_findings`, against the same evidence pool, so the two systems are scored identically. A finding is **supported** when at least one of its citations is valid (the chunk was retrieved and the quote is in it word for word) and the claim shares enough vocabulary with those quotes.

| Metric | Definition |
| --- | --- |
| Pool recall | Expected policy passages that appear in the system's retrieved evidence, over all expected passages (appeal cases). For the baseline this is Recall@5; for the agent it is recall over everything it retrieved. |
| Citation correctness | Valid citations over all citations the model proposed, before any filtering. |
| Unsupported-claim rate, proposed | Findings that are not supported, over all findings the model proposed. This is the model's own error rate. |
| Unsupported-claim rate, shipped | The same, but only over what the user would receive. The agent filters, so it is expected to be near 0 by construction; the baseline ships everything, so it equals the proposed rate. Report both. |
| Answer correctness (appeal) | Share of appeal cases where every expected passage is cited by a supported finding that was shipped. "Cited" means the cited chunk contains the passage, on the expected page. |
| Evidence coverage (appeal) | Expected passages covered, over all expected passages. A softer version of the above. |
| Abstained correctly | Share of no-appeal cases where nothing was shipped. |
| False abstention | Share of appeal cases where nothing was shipped. |
| Model calls per case | The cost side of the comparison. |
| Errors | Cases that failed (for example a model outage). An error is never counted as a correct answer or a correct abstention. |

### Running it

```
python -m insurithm.evals.compare --limit 2     # smoke test
python -m insurithm.evals.compare               # everything
```

This needs `GEMINI_API_KEY` and `GEMINI_MODEL`. The report goes to `evals/reports/comparison_<model>.json` with a Markdown table next to it.

### Status of this evaluation

The harness is finished and tested (`tests/test_compare.py`) with scripted models: one that always cites correctly, one that adds a fabricated finding, and one that only hallucinates. Those tests show that the metrics are computed correctly and that the agent really does filter what the baseline ships. **They are not results.** No real-model comparison has been run yet; the table in the README is empty on purpose.

When it is run, report the model name, the date, and the numbers from the generated file, and keep the caveats below next to them.

## Limits of these numbers

- The policy and the cases were written by the same person, which makes the task easier than it would be on documents from the real world. Before quoting a result, add cases written by someone else and at least one real public policy wording.
- 15 cases is small. One case moves an appeal-case rate by 10 points. One model on one run is one sample; temperature 0 helps but does not make a hosted model fully deterministic.
- Answer correctness is mechanical: it checks that the right passages were cited, not that the claim text is well written or that the argument is persuasive. A human read of the drafts is still needed.
- The overlap test inside the checker is a lexical sanity check. A shipped finding can have a real quotation that does not entail the claim.
- Expected passages are the author's judgement of what a good appeal cites. A system that cites a different but valid clause scores lower on coverage even if its argument is sound.
