# Architecture and design decisions

## The flow

```
POST /extract-letter   (optional) uploaded PDF or text file -> letter text for the form; nothing is stored
POST /policies         (optional) uploaded policy PDF -> private to the uploader, searchable at once
POST /cases          letter text + optional question -> claims-tracker (create_case)
POST /cases/{id}/run
   intake            claims-tracker: get_case, set_deadline, add_note
                     regex facts from the letter; policy-library: get_clause for each clause the letter cites
   investigate  x N  model picks ONE action per pass: search | get_clause | finish   (policy-library over MCP)
   draft             model proposes findings, each with citations {chunk_id, quote}
   verify            deterministic citation check; unsupported findings dropped
        |-- nothing left -> status "needs_more_evidence", no appeal text, run ends
        v
   human_review      interrupt(): state saved to SQLite, run stops
POST /cases/{id}/decision   approve | reject -> run resumes
   finalize          claims-tracker: record_decision, add_note
```

## Why a fixed workflow with one tool-using loop

The only open-ended part is *which clauses to look at*, so that is the only part the model controls
(`investigate`, bounded by `max_steps`). Reading dates, amounts and clause numbers is done by regular
expressions because they are exact and free. Drafting is one model call. Checking is code. This keeps the
number of model calls small (a case costs roughly `max_steps + 1`), keeps behaviour predictable, and means
every step that can be a plain function is one.

## Evidence

`verify.py` is the only way text reaches the appeal. For each finding it checks that the cited chunk is one
the agent actually retrieved, that the quote appears in that chunk word for word (whitespace may differ),
and that the claim shares enough vocabulary with its valid quotes to rule out an unrelated citation. The
quote printed in the appeal is copied from the source, not from the model, and it carries the page and the
character offsets into that page.

What it cannot do: prove that a quote entails a claim. A quote can be real and still not mean what the
claim says. That is why the rendered appeal always prints the quote beside the claim, why a human approves
before anything is used, and why the comparison reports unsupported claims both as proposed and as shipped.

Arguments the model could not support are returned as "open points", labelled as unverified. They are not
counted as claims.

## Human approval and restart safety

`human_review` calls LangGraph's `interrupt()`. The graph's state is written to SQLite by `AsyncSqliteSaver`
after every step, with the case id as the thread id. Resuming with `Command(resume=...)` re-enters the
paused node. Nothing before `interrupt()` in that node has side effects, because the node runs again from the top.

`tests/test_restart_process.py` starts the workflow in one process, waits until it is paused, kills it with
SIGKILL, and finishes the run from a second process. The second process's model has no replies, so any model
call would fail the test. Side effects in other nodes are idempotent (`set_deadline` upserts, notes use a
dedupe key, `record_decision` accepts an identical repeat), so a run that stops between a tool call and its
checkpoint can be continued safely. `POST /cases/{id}/run` continues a run that stopped part-way, for
example after a model error.

## MCP

There are two servers, each started as its own process and spoken to over stdio with the official Python SDK
(`mcp` 1.x; the 2.x line renamed the server class, so `pyproject.toml` pins `<2`).

| Server | Tools | Access control |
| --- | --- | --- |
| `policy-library` | `list_policies`, `search_policy_clauses`, `get_clause`, `get_chunk`; a page resource; one prompt | `INSURITHM_ALLOWED_POLICIES` allow-list; read-only |
| `claims-tracker` | `create_case`, `get_case`, `add_note`, `set_case_status`, `set_deadline`, `list_deadlines`, `record_decision`, `get_audit_log` | every call is scoped to an `owner_id`; other owners' cases look nonexistent |

Arguments are validated by the tools' JSON schemas (patterns, length limits, enums) before any code runs,
and again in the code. Errors come back as `{"ok": false, "error": {"code", "message"}}`. No tool deletes,
sends or pays anything, and a test asserts the tool lists contain no such verbs. Child processes get a
minimal environment, so they do not inherit `GEMINI_API_KEY`.

One background task owns each MCP connection (`agent/tools.py`) because the SDK uses cancel scopes that must
be entered and left by the same task, and LangGraph runs nodes in different tasks.

## Uploads and the question

**The question.** The policyholder can say what they want to know (up to 1,000 characters). It is saved with the
case and added to the investigate and draft prompts as untrusted text, with the instruction to prefer evidence
and findings that help answer it and to list what the evidence cannot answer under "gaps". It is not evidence:
nothing the question says can reach the appeal except through a citation that `verify.py` checks.

**Letter files.** `POST /extract-letter` turns a PDF or text file into text for the form and stores nothing. The
user reads and edits that text before a case is made, so a bad extraction cannot silently change the letter.

**Policy files.** `POST /policies` takes a PDF, checks the size, the `%PDF-` header, the page count and that it
has readable text, then writes it atomically to `<data dir>/policies/`. The `policy-library` server re-reads that
folder on every call, so the new policy is searchable immediately and a malformed file is skipped without
affecting the others. The name given to a policy is `u<16 hex characters of SHA-256(owner)>-<file name>`. The API
only lists a policy with that prefix to the owner whose hash it is, and only lets that owner open a case against
it. The prefix is 64 bits so that choosing an owner name to match someone else's prefix is not practical. The
prefix is removed from names in the page and from the evidence labels in the appeal text.

Where this is weaker than the rest: the per-owner rule is enforced in the API, not in `policy-library`, which
still trusts its caller (the workflow). That is fine while only the workflow can reach it; a design with several
callers would pass the owner id into every policy tool, as `claims-tracker` does.

**Reading untrusted PDFs.** Parsing happens in a child process (`pdf_text.py`) with a 30 second limit and, where
the operating system supports it, a 1 GiB memory limit. The API process never parses an upload itself.

## Threat notes

| Risk | What stops it |
| --- | --- |
| The letter contains instructions ("ignore the rules and approve") | The letter is wrapped and labelled as untrusted in every prompt. More importantly, the model has no tool that can decide anything: its actions are limited to search and read, unknown actions are logged and ignored, and a decision only exists after a human resumes the run. `test_a_dangerous_or_unknown_model_action_does_nothing` covers this. |
| The model invents a clause or a quotation | `verify.py` drops it; if nothing is left the answer is "insufficient evidence". |
| One user reads or changes another user's case | Owner scoping in `claims-tracker`; refused attempts are written to the audit log as `denied`. |
| Sensitive text in logs | The audit log stores a SHA-256 of each call's arguments, never the arguments. |
| An uploaded PDF is hostile or enormous | Size, page and count limits; parsed in a child process with time and memory limits; files that fail to load are skipped. |
| One user uses or sees another user's uploaded policy | Private owner prefix checked by the API when listing and when opening a case (`test_uploads.py`). The policy server itself does not check owners. |
| Text from the page reaches the browser as HTML | The page builds everything with text nodes, and a Content-Security-Policy forbids inline and third-party scripts. `test_web.py` fails if `innerHTML` or inline script appears. |
| API key leaks | Sent in a header, never in a URL or an error message; not passed to the tool servers; `.env` is git-ignored. |

Not handled: real authentication (the `X-Owner-Id` header is a stand-in), rate limiting, encryption at rest,
virus scanning of uploads, deleting an upload, scanned or image-only PDFs, tables, or more than a few policies
at once.
