"use strict";

/* Insurithm web page. Plain JavaScript, no build step.
   Every piece of text that comes from the API (letters, claims, quotes) is added with
   textContent or text nodes, never as HTML, because letters and model output are untrusted. */

const OWNER_KEY = "insurithm.owner";
const RECENT_KEY = "insurithm.recent";

// The sample letter is dated a few days ago, so its appeal window is always still open.
function sampleLetter() {
  const daysAgo = (n) => {
    const d = new Date();
    d.setDate(d.getDate() - n);
    return d;
  };
  const long = new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "long", year: "numeric" });
  const short = new Intl.DateTimeFormat("en-GB", { day: "2-digit", month: "2-digit", year: "numeric" });
  return `Northstar Health Assurance
Date: ${long.format(daysAgo(5))}
Claim No: NHA/CL/2026/00417   Policy No: FFP-889120

Dear Policyholder,

Your claim of Rs. 1,25,000 for the hospitalisation from ${short.format(daysAgo(40))} to ${short.format(daysAgo(37))} has been settled at Rs. 70,000.
We applied a proportionate deduction under clause 4.2 because the room occupied was above the permitted limit.
You may appeal within 30 days of this letter.
`;
}

const DROP_REASONS = {
  no_citations: "No policy passage was cited for this point.",
  no_valid_citation: "The cited passage could not be found in the policy.",
  claim_not_supported_by_quote: "The cited passage does not match what the point says.",
  cites_only_the_letter: "This point rests only on the rejection letter, with no policy passage behind it.",
  missing_claim: "The model's output for this point was unusable.",
  malformed_finding: "The model's output for this point was unusable.",
};

const TOOL_NAMES = {
  create_case: "Case opened",
  get_case: "Case read",
  add_note: "Note added",
  set_case_status: "Status changed",
  set_deadline: "Deadline set",
  list_deadlines: "Deadlines read",
  record_decision: "Decision recorded",
  get_audit_log: "Log read",
};

/* ---------------------------------------------------------------- helpers */
const $ = (selector, root = document) => root.querySelector(selector);

function h(tag, props, ...kids) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") el.className = value;
    else if (key.startsWith("on") && typeof value === "function") el.addEventListener(key.slice(2), value);
    else el.setAttribute(key, value === true ? "" : String(value));
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

const storage = {
  get(key, fallback) {
    try {
      const value = localStorage.getItem(key);
      return value === null ? fallback : value;
    } catch {
      return fallback;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem(key, value);
    } catch {
      /* private window or blocked storage: the page still works without it */
    }
  },
};

function readRecent() {
  try {
    const list = JSON.parse(storage.get(RECENT_KEY, "[]"));
    return Array.isArray(list) ? list.filter((r) => r && typeof r.id === "string") : [];
  } catch {
    return [];
  }
}

function remember(id, letter) {
  const snippet = letter.replace(/\s+/g, " ").trim().slice(0, 90);
  const list = readRecent().filter((r) => r.id !== id);
  list.unshift({ id, snippet, at: Date.now() });
  storage.set(RECENT_KEY, JSON.stringify(list.slice(0, 8)));
}

function parseDate(value) {
  if (value instanceof Date) return value;
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(value));
  return m ? new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3])) : new Date(value);
}

function formatDate(value) {
  const d = parseDate(value);
  if (Number.isNaN(d.getTime())) return String(value);
  return new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short", year: "numeric" }).format(d);
}

function formatDateTime(value) {
  const d = parseDate(value);
  if (Number.isNaN(d.getTime())) return String(value);
  return new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }).format(d);
}

function relativeDays(value) {
  const due = parseDate(value);
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const days = Math.round((due - today) / 86400000);
  if (days === 0) return "today";
  if (days > 0) return `in ${days} day${days === 1 ? "" : "s"}`;
  const ago = -days;
  return `${ago} day${ago === 1 ? "" : "s"} ago`;
}

function policyName(id) {
  return pretty(String(id || "").replace(/^u[0-9a-f]{16}-/, ""));
}

function pretty(id) {
  const text = String(id || "").replace(/[_-]+/g, " ").trim();
  return text.charAt(0).toUpperCase() + text.slice(1);
}

let toastTimer = 0;
function toast(message) {
  const el = $("#toast");
  el.textContent = message;
  el.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), 2600);
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    toast("Copied.");
  } catch {
    toast("Couldn't copy. Select the text and copy it yourself.");
  }
}

/* ---------------------------------------------------------------- talking to the API */
class UiError extends Error {}
class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

const ownerInput = $("#owner");

function cleanOwner(value) {
  return value.trim().replace(/\s+/g, "-").replace(/[^A-Za-z0-9_.@+-]/g, "").slice(0, 64);
}

function owner() {
  const name = cleanOwner(ownerInput.value);
  if (!name) {
    ownerInput.focus();
    throw new UiError("Enter your name at the top of the page first. It keeps your cases private to you.");
  }
  ownerInput.value = name;
  storage.set(OWNER_KEY, name);
  return name;
}

function describe(status, data) {
  const detail = data && data.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail.length) {
    return detail.map((e) => `${(e.loc || []).slice(1).join(".") || "input"}: ${e.msg}`).join("; ");
  }
  if (status === 404) return "That case wasn't found for your name.";
  return `The request failed (${status}).`;
}

async function request(path, { method = "GET", body, raw, auth = true } = {}) {
  // auth: true needs a name, "optional" sends it when there is one, false sends none.
  const headers = {};
  if (auth === true) headers["X-Owner-Id"] = owner();
  else if (auth === "optional" && cleanOwner(ownerInput.value)) headers["X-Owner-Id"] = cleanOwner(ownerInput.value);
  let payload;
  if (raw !== undefined) {
    headers["Content-Type"] = "application/octet-stream";
    payload = raw; // a File: sent as is, the server checks what it really is
  } else if (body !== undefined) {
    headers["Content-Type"] = "application/json";
    payload = JSON.stringify(body);
  }
  let response;
  try {
    response = await fetch(path, { method, headers, body: payload });
  } catch {
    throw new ApiError(0, "Can't reach the server. Check that it is still running.");
  }
  let data = null;
  try {
    data = await response.json();
  } catch {
    /* the body was not JSON */
  }
  if (!response.ok) throw new ApiError(response.status, describe(response.status, data));
  return data;
}

/* ---------------------------------------------------------------- shared pieces */
function mount(view) {
  const root = $("#view");
  root.replaceChildren(view);
  window.scrollTo(0, 0);
  const heading = $("h1", root);
  if (heading) {
    heading.tabIndex = -1;
    heading.focus({ preventScroll: true });
  }
}

function errorView(error, retry, retryLabel = "Try again") {
  const model = error instanceof ApiError && (error.status === 502 || error.status === 503);
  const missing = error instanceof ApiError && error.status === 404;
  const title = model ? "The model didn't answer" : missing ? "Case not found" : "Something went wrong";
  const lead = model
    ? "Your progress is saved, so trying again continues from where it stopped."
    : missing
      ? "This case isn't under the name at the top of the page. Check the name, or go back to start."
      : error instanceof UiError || error instanceof ApiError
        ? error.message
        : "The page hit an unexpected error. Reload it and try again.";
  if (missing) retry = null;
  return h(
    "section",
    null,
    h("h1", null, title),
    h("p", { class: "lede" }, lead),
    model ? h("p", { class: "small muted" }, error.message) : null,
    h(
      "div",
      { class: "actions" },
      retry ? h("button", { class: "btn", type: "button", onclick: retry }, retryLabel) : null,
      h("a", { class: "btn ghost", href: "#/" }, "Back to start"),
    ),
  );
}

function whereLabel(citation) {
  const doc = citation.doc_id || "";
  const parts = [doc.startsWith("letter-") ? "Your letter" : policyName(doc)];
  if (citation.page) parts.push(`page ${citation.page}`);
  if (citation.clause_id) parts.push(`clause ${citation.clause_id}`);
  return parts.join(", ");
}

function findingsList(findings) {
  return h(
    "ol",
    { class: "findings" },
    findings.map((finding) =>
      h(
        "li",
        { class: "finding" },
        h("p", { class: "claim" }, finding.claim),
        (finding.citations || [])
          .filter((c) => c.valid)
          .map((c) =>
            h(
              "figure",
              { class: "quote" },
              h("blockquote", null, String(c.quote || "").replace(/\s+/g, " ")),
              h("figcaption", null, whereLabel(c)),
            ),
          ),
      ),
    ),
  );
}

function droppedBlock(dropped) {
  if (!dropped || !dropped.length) return null;
  const count = dropped.length;
  return h(
    "details",
    null,
    h("summary", null, `${count} point${count === 1 ? "" : "s"} removed by the checker`),
    h(
      "div",
      { class: "inner" },
      h("p", { class: "small muted" }, "The model suggested these, but the policy text doesn't back them up, so they are not in the draft."),
      h(
        "ul",
        { class: "dropped" },
        dropped.map((f) =>
          h("li", null, f.claim || "(empty point)", h("span", null, DROP_REASONS[f.reason] || "It could not be verified.")),
        ),
      ),
    ),
  );
}

function draftBlock(text) {
  if (!text) return null;
  return h(
    "details",
    null,
    h("summary", null, "Appeal draft"),
    h(
      "div",
      { class: "inner" },
      h("pre", { class: "draft" }, text),
      h("button", { class: "btn ghost", type: "button", onclick: () => copyText(text) }, "Copy draft"),
    ),
  );
}

function letterBlock(c) {
  return h(
    "details",
    null,
    h("summary", null, "Original letter"),
    h("div", { class: "inner" }, h("pre", { class: "letter" }, c.case.letter_text)),
  );
}

function activityBlock(id) {
  const body = h("div", { class: "inner" }, h("p", { class: "small muted" }, "Loading"));
  let loaded = false;
  const box = h("details", null, h("summary", null, "Activity log"), body);
  box.addEventListener("toggle", async () => {
    if (!box.open || loaded) return;
    loaded = true;
    try {
      const data = await request(`/cases/${encodeURIComponent(id)}/audit`);
      const rows = (data.entries || []).map((e) =>
        h(
          "tr",
          null,
          h("td", null, formatDateTime(e.ts)),
          h("td", null, TOOL_NAMES[e.tool] || pretty(e.tool)),
          h("td", null, e.status === "ok" ? "Done" : e.status === "denied" ? "Refused" : `Failed${e.error_code ? ` (${e.error_code})` : ""}`),
          h("td", null, e.owner_id),
        ),
      );
      body.replaceChildren(
        h("p", { class: "small muted" }, "Every tool call made on this case. The log keeps a fingerprint of the inputs, not the text itself."),
        h(
          "table",
          { class: "log" },
          h("thead", null, h("tr", null, ["When", "What", "Result", "By"].map((t) => h("th", { scope: "col" }, t)))),
          h("tbody", null, rows),
        ),
      );
    } catch (error) {
      loaded = false;
      body.replaceChildren(h("p", { class: "small" }, error.message));
    }
  });
  return box;
}

function questionBlock(c) {
  const question = c.case.question;
  if (!question) return null;
  return h("div", { class: "ask" }, h("span", { class: "ask-label" }, "Your question"), h("p", null, question));
}

function factsList(c) {
  const f = c.workflow && c.workflow.facts;
  const items = [["Policy", policyName(c.case.policy_id)]];
  if (f) {
    const names = { rejected: "Rejected", delayed: "Delayed", short_paid: "Paid short" };
    if ((f.decision_types || []).length) items.push(["Decision", f.decision_types.map((d) => names[d] || d).join(", ")]);
    const claim = (f.facts || []).find((x) => x.kind === "claim_number");
    if (claim) items.push(["Claim", claim.value]);
    if ((f.clause_refs || []).length) items.push(["Clauses cited", f.clause_refs.join(", ")]);
  }
  for (const d of c.deadlines || []) items.push([d.label, `${formatDate(d.due_date)} (${relativeDays(d.due_date)})`]);
  if (!items.length) return null;
  return h("ul", { class: "facts" }, items.map(([label, value]) => h("li", null, h("span", null, label, " "), h("b", null, value))));
}

/* ---------------------------------------------------------------- home */
let onOwnerChange = null; // the start page re-reads the policy list when the name changes

function homeView(initialPolicies) {
  let policies = initialPolicies;
  const question = h("textarea", {
    id: "question", class: "short", maxlength: 1000, spellcheck: "false",
    placeholder: "For example: Why was my claim cut to Rs. 70,000? Can I still appeal, and by when?",
  });
  const letter = h("textarea", { id: "letter", maxlength: 20000, spellcheck: "false", placeholder: "Paste the text of the letter here, or upload the file." });
  const count = h("span", { class: "small muted" }, "0 / 20,000");
  const fileNote = h("span", { class: "small muted", role: "status" });
  const problem = h("div", { class: "error", role: "alert", hidden: true });
  const go = h("button", { class: "btn", type: "submit", disabled: true }, "Review letter");
  const sample = h("button", { class: "btn ghost", type: "button" }, "Use a sample letter");
  const picker = h("select", { class: "text", id: "policy" });
  const letterFile = h("input", { type: "file", accept: ".pdf,.txt,application/pdf,text/plain", hidden: true, "aria-label": "Letter file" });
  const policyFile = h("input", { type: "file", accept: ".pdf,application/pdf", hidden: true, "aria-label": "Policy file" });
  const letterButton = h("button", { class: "link small", type: "button" }, "Upload a PDF or text file");
  const policyButton = h("button", { class: "btn ghost small-btn", type: "button" }, "Upload your own policy (PDF)");

  function fail(message) {
    problem.textContent = message;
    problem.hidden = false;
  }

  function sync() {
    count.textContent = `${letter.value.length.toLocaleString("en-US")} / 20,000`;
    go.disabled = letter.value.trim().length < 20;
  }

  function showPolicies(selectId) {
    const keep = selectId || picker.value;
    picker.replaceChildren(...policies.map((p) => h("option", { value: p.policy_id }, `${pretty(p.title || p.policy_id)}${p.mine ? " (yours)" : ""}`)));
    if (policies.some((p) => p.policy_id === keep)) picker.value = keep;
  }

  async function refreshPolicies(selectId) {
    try {
      const list = await request("/policies", { auth: "optional" });
      if (Array.isArray(list)) policies = list;
    } catch {
      /* keep the list we have */
    }
    showPolicies(selectId);
  }
  onOwnerChange = () => refreshPolicies();

  async function loadLetterFile(file) {
    problem.hidden = true;
    fileNote.textContent = `Reading ${file.name}`;
    try {
      const result = await request("/extract-letter", { method: "POST", raw: file });
      letter.value = result.text;
      sync();
      fileNote.textContent = `Loaded from ${file.name}${result.truncated ? ", cut to the first 20,000 characters" : ""}. Check the text before you continue.`;
    } catch (error) {
      fileNote.textContent = "";
      fail(error.message);
    }
  }

  async function loadPolicyFile(file) {
    problem.hidden = true;
    policyButton.disabled = true;
    policyButton.textContent = "Reading policy";
    try {
      const made = await request(`/policies?name=${encodeURIComponent(file.name)}`, { method: "POST", raw: file });
      await refreshPolicies(made.policy_id);
      toast("Policy added.");
    } catch (error) {
      fail(error.message);
    } finally {
      policyButton.disabled = false;
      policyButton.textContent = "Upload your own policy (PDF)";
    }
  }

  function pickFile(input, handler) {
    input.addEventListener("change", () => {
      const file = input.files && input.files[0];
      input.value = ""; // so choosing the same file again still fires
      if (file) handler(file);
    });
  }
  pickFile(letterFile, loadLetterFile);
  pickFile(policyFile, loadPolicyFile);
  letterButton.addEventListener("click", () => letterFile.click());
  policyButton.addEventListener("click", () => policyFile.click());

  letter.addEventListener("input", sync);
  letter.addEventListener("dragover", (event) => {
    if (event.dataTransfer && Array.from(event.dataTransfer.types).includes("Files")) {
      event.preventDefault();
      letter.classList.add("drop");
    }
  });
  letter.addEventListener("dragleave", () => letter.classList.remove("drop"));
  letter.addEventListener("drop", (event) => {
    letter.classList.remove("drop");
    const file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
    if (file) {
      event.preventDefault();
      loadLetterFile(file);
    }
  });
  sample.addEventListener("click", () => {
    letter.value = sampleLetter();
    fileNote.textContent = "";
    sync();
    letter.focus();
  });
  showPolicies();

  const form = h(
    "form",
    {
      novalidate: true,
      async onsubmit(event) {
        event.preventDefault();
        problem.hidden = true;
        go.disabled = true;
        go.textContent = "Opening case";
        try {
          const body = { letter_text: letter.value.trim(), question: question.value.trim() };
          if (picker.value) body.policy_id = picker.value;
          const made = await request("/cases", { method: "POST", body });
          remember(made.case_id, question.value.trim() || letter.value);
          pendingRun = made.case_id;
          location.hash = `#/case/${made.case_id}`;
        } catch (error) {
          fail(error.message);
          go.textContent = "Review letter";
          sync();
        }
      },
    },
    h("div", { class: "block" },
      h("label", { class: "field", for: "question" }, "What's the problem, or what do you want to know?"),
      question,
      h("p", { class: "small muted note-top" }, "Optional, but it helps the review focus on what matters to you."),
    ),
    h("div", { class: "block" },
      h("div", { class: "fieldrow" }, h("label", { class: "field", for: "letter" }, "Rejection or settlement letter"), letterButton),
      letter,
      h("div", { class: "meta-row" }, fileNote, count),
      letterFile,
    ),
    h("div", { class: "block" },
      h("label", { class: "field", for: "policy" }, "Policy to check against"),
      picker,
      h("div", { class: "policy-row" }, policyButton, h("span", { class: "small muted" }, "Only you can see policies you upload.")),
      policyFile,
    ),
    problem,
    h("div", { class: "actions" }, go, sample),
  );

  const recent = readRecent();
  const recentBlock = recent.length
    ? h(
        "section",
        { class: "recent" },
        h(
          "div",
          { class: "recent-head" },
          h("h2", null, "Recent cases"),
          h(
            "button",
            {
              class: "link small",
              type: "button",
              onclick() {
                storage.set(RECENT_KEY, "[]");
                route();
              },
            },
            "Clear list",
          ),
        ),
        h(
          "ul",
          { class: "rows" },
          recent.map((r) =>
            h(
              "li",
              null,
              h("a", { href: `#/case/${encodeURIComponent(r.id)}` }, h("span", { class: "snippet" }, r.snippet || r.id), h("time", null, formatDate(new Date(r.at)))),
            ),
          ),
        ),
      )
    : null;

  return h(
    "section",
    null,
    h("h1", null, "Appeals that quote the policy."),
    h("p", { class: "lede" }, "Tell us what happened and paste or upload the rejection letter. Every point in the appeal cites the exact passage and page. If the policy doesn't support a point, it is left out, and nothing is final until you approve it."),
    form,
    recentBlock,
  );
}

/* ---------------------------------------------------------------- case pages */
function runningView() {
  return h(
    "section",
    { role: "status" },
    h("h1", null, "Reviewing the letter"),
    h("p", { class: "lede tight" }, "Reading the letter, searching the policy and checking every quote against it. This can take a minute."),
    h("div", { class: "bar", "aria-hidden": "true" }),
    h("p", { class: "small muted" }, "You can leave this page. The review keeps going and you can come back to it from Recent cases."),
  );
}

function readyView(c, stopped) {
  const id = c.case.case_id;
  return h(
    "section",
    null,
    h("h1", null, stopped ? "This review stopped part-way" : "Ready to review"),
    h("p", { class: "lede" }, stopped ? "The saved progress is kept, so continuing picks up where it stopped." : "The letter is saved. Start the review to search the policy and draft the appeal."),
    questionBlock(c),
    h("div", { class: "actions before-group" }, h("button", { class: "btn", type: "button", onclick: () => runCase(id) }, stopped ? "Continue review" : "Start review")),
    h("div", { class: "group" }, letterBlock(c), activityBlock(id)),
  );
}

function reviewView(c) {
  const id = c.case.case_id;
  const review = c.workflow.review;
  const supported = review.findings || [];
  const comment = h("textarea", { class: "short", id: "comment", maxlength: 1000, placeholder: "Add a note for the record" });
  const problem = h("div", { class: "error", role: "alert", hidden: true });
  const approve = h("button", { class: "btn", type: "button" }, "Approve");
  const reject = h("button", { class: "btn ghost", type: "button" }, "Reject");

  async function decide(choice) {
    problem.hidden = true;
    approve.disabled = reject.disabled = true;
    const pressed = choice === "approve" ? approve : reject;
    pressed.textContent = "Saving";
    try {
      await request(`/cases/${encodeURIComponent(id)}/decision`, {
        method: "POST",
        body: { decision: choice, reviewer: owner(), comment: comment.value.trim() },
      });
      toast(choice === "approve" ? "Approved." : "Rejected.");
      await showCase(id, nav);
    } catch (error) {
      problem.textContent = error.message;
      problem.hidden = false;
      approve.disabled = reject.disabled = false;
      approve.textContent = "Approve";
      reject.textContent = "Reject";
    }
  }
  approve.addEventListener("click", () => decide("approve"));
  reject.addEventListener("click", () => decide("reject"));

  return h(
    "section",
    null,
    h("span", { class: "status" }, "Waiting for your decision"),
    h("h1", null, "Review the draft"),
    h("p", { class: "lede" }, "Check each point against the policy text under it. Approve only what you would put your name to."),
    questionBlock(c),
    factsList(c),
    h("h2", null, `${supported.length} point${supported.length === 1 ? "" : "s"} backed by the policy`),
    findingsList(supported),
    h("div", { class: "section group" }, droppedBlock(review.dropped_findings), draftBlock(review.appeal_text), letterBlock(c)),
    h(
      "form",
      { class: "decision", onsubmit: (e) => e.preventDefault() },
      h("h2", null, "Your decision"),
      h("label", { class: "field", for: "comment" }, "Comment (optional)"),
      comment,
      problem,
      h("div", { class: "actions" }, approve, reject),
      h("p", { class: "small muted note" }, "This records your decision. Nothing is sent to the insurer."),
    ),
    h("div", { class: "section group" }, activityBlock(id)),
  );
}

function doneView(c) {
  const id = c.case.case_id;
  const wf = c.workflow;
  const verification = wf.verification || { supported: [], unsupported: [] };

  if (wf.outcome === "insufficient_evidence" || c.case.status === "needs_more_evidence") {
    const note = (c.notes || []).find((n) => /^Insufficient evidence/.test(n.text));
    const gaps = note && note.text.includes("Unsupported: ") ? note.text.split("Unsupported: ")[1].split("; ").filter(Boolean) : [];
    return h(
      "section",
      null,
      h("span", { class: "status" }, "No draft made"),
      h("h1", null, "Not enough evidence"),
      h("p", { class: "lede" }, "The policy text doesn't support an appeal point for this letter, so there is no draft. Nothing was made up to fill the gap."),
      questionBlock(c),
      gaps.length
        ? h("div", { class: "section first" }, h("h2", null, "What the policy doesn't cover"), h("ul", { class: "dropped strong" }, gaps.map((g) => h("li", null, g))))
        : null,
      h("div", { class: "section group" }, droppedBlock(verification.unsupported), letterBlock(c), activityBlock(id)),
      h("div", { class: "actions" }, h("a", { class: "btn", href: "#/" }, "Review another letter")),
    );
  }

  const decision = c.decision;
  const approved = decision && decision.decision === "approved";
  return h(
    "section",
    null,
    h("span", { class: "status solid" }, approved ? "Approved" : "Rejected"),
    h("h1", null, approved ? "Draft approved" : "Draft rejected"),
    decision ? h("p", { class: "record" }, `${decision.reviewer} ${decision.decision} this draft on ${formatDate(decision.created_at)}.`) : null,
    decision && decision.comment ? h("p", { class: "muted" }, `"${decision.comment}"`) : null,
    approved ? h("p", { class: "lede" }, "The draft is ready to copy. Insurithm doesn't send anything, so you send it to the insurer yourself.") : h("p", { class: "lede" }, "The draft was not used. The decision is recorded in the activity log."),
    questionBlock(c),
    factsList(c),
    h("h2", null, `${verification.supported.length} point${verification.supported.length === 1 ? "" : "s"} backed by the policy`),
    findingsList(verification.supported),
    h("div", { class: "section group" }, droppedBlock(verification.unsupported), draftBlock(wf.appeal_text), letterBlock(c)),
    h("div", { class: "section group" }, activityBlock(id)),
    h("div", { class: "actions" }, h("a", { class: "btn ghost", href: "#/" }, "Review another letter")),
  );
}

/* ---------------------------------------------------------------- routing */
let nav = 0;
let pendingRun = null; // a case that was just created and should start by itself

async function runCase(id, token = nav) {
  mount(runningView());
  try {
    await request(`/cases/${encodeURIComponent(id)}/run`, { method: "POST" });
  } catch (error) {
    if (token === nav) mount(errorView(error, () => runCase(id, nav)));
    return;
  }
  if (token === nav) await showCase(id, token);
}

async function showCase(id, token) {
  const c = await request(`/cases/${encodeURIComponent(id)}`);
  if (token !== nav) return;
  const status = c.workflow.status;
  if (status === "not_started") {
    if (pendingRun === id) {
      pendingRun = null;
      return runCase(id, token);
    }
    return mount(readyView(c, false));
  }
  if (status === "incomplete") return mount(readyView(c, true));
  if (status === "awaiting_review") return mount(reviewView(c));
  return mount(doneView(c));
}

async function showHome(token) {
  let policies = [];
  try {
    policies = await request("/policies", { auth: "optional" });
  } catch {
    /* the form still works; the server picks the policy */
  }
  if (token === nav) mount(homeView(Array.isArray(policies) ? policies : []));
}

async function route() {
  const token = ++nav;
  onOwnerChange = null;
  const match = /^#\/case\/([A-Za-z0-9_-]{3,64})$/.exec(location.hash);
  try {
    if (match) await showCase(match[1], token);
    else await showHome(token);
  } catch (error) {
    if (token === nav) mount(errorView(error, () => route()));
  }
}

/* ---------------------------------------------------------------- start */
ownerInput.value = storage.get(OWNER_KEY, "");
ownerInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") ownerInput.blur();
});
ownerInput.addEventListener("change", () => {
  ownerInput.value = cleanOwner(ownerInput.value);
  storage.set(OWNER_KEY, ownerInput.value);
  if (location.hash.startsWith("#/case/")) route();
  else if (onOwnerChange) onOwnerChange();
});
window.addEventListener("hashchange", route);

request("/config", { auth: false })
  .then((config) => {
    $("#demo").hidden = !(config && config.demo);
  })
  .catch(() => {});

route();
