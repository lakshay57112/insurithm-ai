"""Deterministic fact extraction from a rejection or settlement letter.

Dates, amounts, clause numbers and appeal windows are pulled out with plain
regular expressions rather than by a language model. They are exact, they cost
nothing, and each fact keeps the span of the letter it came from, so
`letter[fact.start:fact.end] == fact.text` always holds and a reviewer can see
where it came from.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from typing import Any, Iterable

MONTHS = {
    name: i
    for i, names in enumerate(
        [
            ("jan", "january"),
            ("feb", "february"),
            ("mar", "march"),
            ("apr", "april"),
            ("may",),
            ("jun", "june"),
            ("jul", "july"),
            ("aug", "august"),
            ("sep", "sept", "september"),
            ("oct", "october"),
            ("nov", "november"),
            ("dec", "december"),
        ],
        start=1,
    )
    for name in names
}
_MONTH_RE = "|".join(sorted(MONTHS, key=len, reverse=True))

_NUM = r"\d+(?:\.\d+)*"
_CLAUSE_RE = re.compile(
    rf"\b(?:clauses?|sections?|exclusions?|conditions?|paragraphs?|para)\s+(?:no\.?\s*)?"
    rf"({_NUM}(?:\s*(?:,|and|&|/)\s*{_NUM})*)",
    re.IGNORECASE,
)
_ANNEXURE_RE = re.compile(r"\bannexure\s+([A-Z])\b", re.IGNORECASE)
_AMOUNT_RE = re.compile(
    r"(?:\bRs\.?|\bINR|₹)\s*([\d,]*\d(?:\.\d+)?)(?:\s*/-)?(?:\s*(lakhs?|lacs?|crores?))?",
    re.IGNORECASE,
)
_DATE_PATTERNS = [
    # 12 September 2026, 12th Sep, 2026
    re.compile(rf"(?<!\d)(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_RE})\.?,?\s+(\d{{4}})(?!\d)", re.IGNORECASE),
    # September 12, 2026
    re.compile(rf"\b({_MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})(?!\d)", re.IGNORECASE),
    # 2026-09-12
    re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)"),
    # 12/09/2026, 12-09-2026, 12.09.2026 (day first, the Indian convention)
    re.compile(r"(?<!\d)(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})(?!\d)"),
]
_LETTER_DATE_LABEL = re.compile(r"(?:\bdated?|\bdt\.?)\s*[:\-]?\s*$", re.IGNORECASE)
_WINDOW_RE = re.compile(
    r"\bwithin\s+(\d{1,3})\s+(calendar\s+|working\s+|business\s+)?days\b", re.IGNORECASE
)
_REF_RE = re.compile(
    r"\b(claim|policy)\s*(?:no\.?|number|ref(?:erence)?\.?|id)\s*[:#\-]?\s*([A-Z0-9][A-Z0-9/\-]{3,})",
    re.IGNORECASE,
)
_DECISION_PATTERNS = {
    "rejected": re.compile(
        r"\b(reject(?:ed|ion)?|repudiat\w*|denied|denial|declin\w*|not\s+payable|inadmissible)\b", re.IGNORECASE
    ),
    "delayed": re.compile(
        r"\b(delay\w*|pending|yet\s+to\s+be\s+(?:processed|settled)|not\s+yet\s+settled|under\s+(?:review|investigation))\b",
        re.IGNORECASE,
    ),
    "short_paid": re.compile(
        r"\b(short[- ]?paid|partial(?:ly)?\s+(?:settle\w*|paid|approved)|deduct\w*|disallow\w*|"
        r"proportionate\w*|co-?pay\w*|reduced)\b",
        re.IGNORECASE,
    ),
}


@dataclass(frozen=True)
class Fact:
    kind: str  # clause_ref | annexure_ref | amount | date | window_days | decision_type | claim_number | policy_number
    value: str  # normalised: "4.2", "125000", "2026-09-12", "30", "rejected"
    text: str  # exact text in the letter
    start: int
    end: int
    detail: str | None = None  # e.g. "letter_date", "working"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LetterFacts:
    facts: list[Fact]

    def of_kind(self, kind: str) -> list[Fact]:
        return [f for f in self.facts if f.kind == kind]

    def _unique(self, kind: str) -> list[str]:
        seen: dict[str, None] = {}
        for f in self.of_kind(kind):
            seen.setdefault(f.value, None)
        return list(seen)

    @property
    def clause_refs(self) -> list[str]:
        return self._unique("clause_ref")

    @property
    def decision_types(self) -> list[str]:
        return self._unique("decision_type")

    @property
    def letter_date(self) -> str | None:
        labelled = [f for f in self.of_kind("date") if f.detail == "letter_date"]
        return labelled[0].value if labelled else None

    @property
    def appeal_deadline(self) -> tuple[str, str] | None:
        """(ISO date, explanation) when the letter gives both its own date and a calendar-day window.

        Counted from the letter date, not the date of receipt, so it is the earliest the
        window could close and therefore the safe date to work to.
        """
        if self.letter_date is None:
            return None
        windows = [f for f in self.of_kind("window_days") if f.detail in (None, "calendar")]
        if not windows:
            return None
        days = int(windows[0].value)
        due = date.fromisoformat(self.letter_date) + timedelta(days=days)
        return due.isoformat(), f"{days} days from the letter date {self.letter_date} (receipt date not used)"

    def to_dict(self) -> dict[str, Any]:
        return {
            "facts": [f.to_dict() for f in self.facts],
            "clause_refs": self.clause_refs,
            "decision_types": self.decision_types,
            "letter_date": self.letter_date,
        }


def _valid_date(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _parse_date(pattern_index: int, groups: tuple[str, ...]) -> date | None:
    if pattern_index == 0:
        d, month, y = groups
        return _valid_date(int(y), MONTHS[month.lower()], int(d))
    if pattern_index == 1:
        month, d, y = groups
        return _valid_date(int(y), MONTHS[month.lower()], int(d))
    if pattern_index == 2:
        y, m, d = groups
        return _valid_date(int(y), int(m), int(d))
    a, b, y = groups  # day first; fall back to month first only if day first is impossible
    return _valid_date(int(y), int(b), int(a)) or _valid_date(int(y), int(a), int(b))


def _amount_value(number: str, unit: str | None) -> str:
    value = float(number.replace(",", ""))
    if unit:
        u = unit.lower()
        value *= 10_000_000 if u.startswith("crore") else 100_000
    return str(int(value)) if value == int(value) else f"{value:.2f}"


def _overlaps(span: tuple[int, int], used: Iterable[tuple[int, int]]) -> bool:
    return any(span[0] < e and s < span[1] for s, e in used)


def extract_facts(letter: str) -> LetterFacts:
    facts: list[Fact] = []

    for m in _CLAUSE_RE.finditer(letter):
        for ref in re.findall(_NUM, m.group(1)):
            facts.append(Fact("clause_ref", ref, m.group(0), m.start(), m.end()))
    for m in _ANNEXURE_RE.finditer(letter):
        facts.append(Fact("annexure_ref", m.group(1).upper(), m.group(0), m.start(), m.end()))

    for m in _AMOUNT_RE.finditer(letter):
        facts.append(Fact("amount", _amount_value(m.group(1), m.group(2)), m.group(0), m.start(), m.end()))

    used: list[tuple[int, int]] = []
    for idx, pattern in enumerate(_DATE_PATTERNS):
        for m in pattern.finditer(letter):
            span = (m.start(), m.end())
            if _overlaps(span, used):
                continue
            parsed = _parse_date(idx, m.groups())
            if parsed is None:
                continue
            used.append(span)
            label = bool(_LETTER_DATE_LABEL.search(letter[max(0, m.start() - 12) : m.start()]))
            facts.append(
                Fact("date", parsed.isoformat(), m.group(0), m.start(), m.end(), "letter_date" if label else None)
            )

    for m in _WINDOW_RE.finditer(letter):
        kind = (m.group(2) or "").strip().lower()
        detail = "working" if kind in ("working", "business") else "calendar"
        facts.append(Fact("window_days", m.group(1), m.group(0), m.start(), m.end(), detail))

    for m in _REF_RE.finditer(letter):
        value = m.group(2)
        if any(ch.isdigit() for ch in value):
            facts.append(Fact(f"{m.group(1).lower()}_number", value, m.group(0), m.start(), m.end()))

    for label, pattern in _DECISION_PATTERNS.items():
        m = pattern.search(letter)
        if m:
            facts.append(Fact("decision_type", label, m.group(0), m.start(), m.end()))

    facts.sort(key=lambda f: (f.start, f.kind))
    return LetterFacts(facts)
