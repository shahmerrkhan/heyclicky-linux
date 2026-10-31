"""Answer length: four levels picked with the input bar's buttons, or heard in a voice question.

  - None (no pick): exactly today's length, nothing is added to any prompt.
  - "extra_short", "short", "medium", "long": one line at the very end of the
    system prompt (Vision, Write) or `--append-system-prompt` (Code: only the
    closing report). Instructions only: no token cap, no cutting afterwards.

A pick lasts one question. Typed text is never searched for length words.

Voice: trigger phrases (TRIGGERS) only wake the check; they never set a
length on their own. The check, in order:
  a. the fast model judges every trigger in one call (FAST_TIMEOUT_S);
  b. if that fails (busy, limits, timeout, unreadable reply), the strong model
     (STRONG_TIMEOUT_S); a model remembered as daily-limited is skipped;
  c. if both fail, simple built-in rules: "counts", "doesn't count" or "unsure";
  d. an unsure trigger counts in Vision only (Write and Code: normal length).
Two different levels left afterwards means none is used.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Literal, Protocol

from .gemini import GeminiErrorKind, GeminiRequest, classify_error

logger = logging.getLogger(__name__)

Length = Literal["extra_short", "short", "medium", "long"]
LEVELS: tuple[Length, ...] = ("extra_short", "short", "medium", "long")
LABELS: dict[str, str] = {"extra_short": "extra short", "short": "short", "medium": "medium", "long": "long"}

FAST_TIMEOUT_S = 3.0
STRONG_TIMEOUT_S = 5.0

# --- sizes and the instruction lines ------------------------------------------------

_SIZES: dict[str, dict[str, str]] = {
    "vision": {
        "extra_short": "1 sentence, about 25 words at most",
        "short": "3 to 5 sentences, about 60 to 110 words",
        "medium": "2 short paragraphs, about 120 to 180 words",
        "long": "3 to 6 short paragraphs, about 200 to 400 words",
    },
    "writing": {
        "extra_short": "1 to 2 sentences, about 35 words at most",
        "short": "3 to 6 sentences, about 60 to 120 words",
        "medium": "2 short paragraphs, about 120 to 170 words",
        "long": "2 to 4 paragraphs, about 200 to 300 words",
    },
    "code": {
        "extra_short": "1 sentence, about 25 words at most",
        "short": "2 to 4 sentences, about 40 to 80 words",
        "medium": "1 to 2 short paragraphs, about 100 to 160 words",
        "long": "3 to 5 short paragraphs, about 200 to 350 words",
    },
}


def instruction(length: str | None, mode: str) -> str:
    """The length line for this question ("" when no length was picked)."""
    if length is None:
        return ""
    size = _SIZES[mode][length]
    label = LABELS[length]
    if mode == "code":
        return (
            f"When you have finished, write your final report (your message after the last tool call) "
            f"at this length: {label}, {size}. This applies only to that final report. Never let it limit "
            "the work itself: your edits, tool use and number of turns stay exactly as the task needs."
        )
    if mode == "writing":
        return (
            f"\n\nlength for this draft: {label}, {size}. this overrides the default length rule above "
            "and any length words in the user's request.\n"
        )
    return (
        f"\n\nlength for this answer: {label}, {size}. this overrides the default length rule above "
        "and any length words in the user's message. the [POINT] tag at the end doesn't count.\n"
    )


# --- trigger phrases ----------------------------------------------------------------

TRIGGERS: dict[str, tuple[str, ...]] = {
    "extra_short": (
        "in one sentence", "in a sentence", "one sentence", "one-sentence", "one liner", "one-liner",
        "in one line", "in a line", "one line", "tl;dr", "tldr", "in a word", "in a few words",
        "very short", "super short", "really short", "extra short", "ultra short", "very brief",
        "super brief", "really brief", "as short as possible", "shortest possible", "the shortest",
        "bare minimum", "in a nutshell", "just the answer", "just the answer please", "no explanation",
    ),
    "short": (
        "briefly", "in brief", "in short", "keep it short", "keep it brief", "keep it concise",
        "be concise", "concisely", "short and sweet", "short and simple", "short answer", "short version",
        "short summary", "quick answer", "quick summary", "quick rundown", "quick overview",
        "quick explanation", "quick version", "the short version", "a few sentences",
        "a couple of sentences", "two or three sentences", "in a paragraph", "one paragraph",
        "a short paragraph", "sum it up", "summarize it", "give me a summary", "the gist",
        "the gist of it", "high level", "high-level", "just the highlights", "just the basics",
        "the basics", "without going into detail", "don't go into detail", "no need for detail",
        "not too detailed",
    ),
    "medium": (
        "some detail", "a bit of detail", "a little detail", "a bit more detail", "a little more detail",
        "more detail", "a bit more", "a little more", "explain a bit", "explain a little", "expand a bit",
        "expand a little", "a couple of paragraphs", "two paragraphs", "a paragraph or two",
        "two or three paragraphs", "medium length", "medium answer", "medium detail", "moderate detail",
        "moderately detailed", "somewhat detailed", "fairly detailed", "reasonably detailed",
        "a decent amount of detail", "a decent explanation", "middle ground", "not too long",
        "not too short",
    ),
    "long": (
        "in detail", "in depth", "in-depth", "detailed", "a detailed explanation", "very detailed",
        "super detailed", "really detailed", "as detailed as possible", "in great detail",
        "in full detail", "full detail", "go into detail", "go into depth", "go deep", "go deeper",
        "explain fully", "fully explain", "explain in full", "explain thoroughly", "thoroughly",
        "comprehensive", "comprehensively", "extensively", "exhaustive", "exhaustively", "elaborate",
        "elaborate on", "expand on", "expand on that", "tell me everything", "the full story",
        "full explanation", "long answer", "long version", "long explanation", "lengthy", "at length",
        "as long as needed", "take your time", "walk me through", "walk through", "step by step",
        "step-by-step", "break it down", "break it all down", "go through it all", "cover everything",
        "don't leave anything out", "leave nothing out", "don't hold back", "in full",
    ),
}


def _phrase_pattern(phrase: str) -> str:
    """Hyphen or space between words; "tl;dr" also as "tldr" / "tl dr"; "don't" with or without '."""
    words = []
    for word in re.split(r"[\s-]+", phrase.lower()):
        part = re.escape(word).replace(";", r"[\s;:-]*").replace("'", "['’]?")
        words.append(part)
    return r"(?<!\w)" + r"[\s-]+".join(words) + r"(?!\w)"


def _compile_triggers() -> list[tuple[re.Pattern[str], str]]:
    seen: set[str] = set()
    compiled: list[tuple[re.Pattern[str], str]] = []
    for level, phrases in TRIGGERS.items():
        for phrase in phrases:
            pattern = _phrase_pattern(phrase)
            if pattern in seen:  # "in depth" / "in-depth" are the same pattern
                continue
            seen.add(pattern)
            compiled.append((re.compile(pattern, re.IGNORECASE), level))
    return compiled


_TRIGGER_PATTERNS = _compile_triggers()


@dataclass(frozen=True)
class Trigger:
    text: str  # as heard
    level: str
    start: int
    end: int


def find_triggers(transcript: str) -> list[Trigger]:
    """Every trigger phrase in the transcript; where two overlap, the longer one wins."""
    found: list[Trigger] = []
    for pattern, level in _TRIGGER_PATTERNS:
        for match in pattern.finditer(transcript):
            found.append(Trigger(match.group(0), level, match.start(), match.end()))
    found.sort(key=lambda t: (-(t.end - t.start), t.start))
    kept: list[Trigger] = []
    for trigger in found:
        if all(trigger.end <= k.start or trigger.start >= k.end for k in kept):
            kept.append(trigger)
    return sorted(kept, key=lambda t: t.start)


# --- step c: simple built-in rules --------------------------------------------------

COUNTS, NO, UNSURE = "counts", "no", "unsure"

_SENTENCE_END = re.compile(r"[.!?]")
# Dictated content: "tell sam ...", "say that ...", "write that ...", "let sam know ...".
_DICTATION = re.compile(
    r"\b(?:(?:tell|ask|text|email)\s+(?!me\b|us\b)\w+|say|write|reply|mention|let\s+\w+\s+know)\b",
    re.IGNORECASE,
)
# An instruction to Clicky a few words before the trigger: "explain this in detail".
_INSTRUCTION_CUE = re.compile(
    r"\b(?:explain|answer|tell\s+me|give\s+me|keep\s+it|make\s+it|can\s+you|could\s+you)\b(?:\W+\w+){0,4}\W*$",
    re.IGNORECASE,
)
_QUOTES = {'"': '"”', "“": '"”', "”": '"”'}


def _inside_quotes(text: str, position: int) -> bool:
    i = 0
    while i < len(text):
        closers = _QUOTES.get(text[i])
        if closers is None:
            i += 1
            continue
        end = next((j for j in range(i + 1, len(text)) if text[j] in closers), None)
        if end is None:
            return False
        if i < position < end:
            return True
        i = end + 1
    return False


def rule_verdict(transcript: str, trigger: Trigger) -> str:
    """COUNTS, NO or UNSURE for one trigger. "Doesn't count" beats "counts"."""
    sentence_start = max((m.end() for m in _SENTENCE_END.finditer(transcript, 0, trigger.start)), default=0)
    before = transcript[sentence_start : trigger.start]
    after = transcript[trigger.end :]
    if _inside_quotes(transcript, trigger.start) or _DICTATION.search(before):
        return NO
    at_start = not re.search(r"\w", transcript[: trigger.start])
    at_end = not re.search(r"\w", re.sub(r"\bplease\b", "", after, flags=re.IGNORECASE))
    please_next = re.match(r"\W*please\b", after, re.IGNORECASE) is not None
    if at_start or at_end or please_next or _INSTRUCTION_CUE.search(before):
        return COUNTS
    return UNSURE


# --- steps a and b: a Gemini model judges ------------------------------------------

_TARGETS = {
    "vision": "how long clicky's answer should be",
    "writing": (
        "how long the message draft clicky writes should be. words the user wants said in the message "
        "itself are content: \"tell sam i'll be brief\" is content"
    ),
    "code": (
        "how long claude code's closing report should be. phrases about the coding work itself "
        "(what to change, build, cover or test) are content: \"break it down into functions\" is content"
    ),
}

JUDGE_SYSTEM_PROMPT = """\
you check length words in a speech-to-text transcript of a user talking to an assistant called clicky.
for each numbered phrase, decide whether the user is telling clicky {target} ("instruction"), or whether the phrase is part of what they are saying, dictating or asking about ("content").
reply with one line per phrase and nothing else, exactly like:
1: instruction
2: content
"""

_VERDICT_LINE = re.compile(r"^\W*(\d+)\s*[:.)\-]\s*\W*(instruction|content)\b", re.IGNORECASE | re.MULTILINE)


class TextModel(Protocol):
    async def generate_text(
        self, request: GeminiRequest, system_prompt: str, model: str, thinking_level: str
    ) -> str: ...


class DailyLimits(Protocol):
    def is_daily_limited(self, model: str) -> bool: ...
    def mark_daily_limited(self, model: str) -> None: ...


def _judge_prompt(transcript: str, triggers: list[Trigger]) -> str:
    lines = [f"transcript:\n{transcript}\n", "phrases:"]
    for number, trigger in enumerate(triggers, start=1):
        context = transcript[max(0, trigger.start - 40) : trigger.end + 40].replace("\n", " ")
        lines.append(f'{number}. "{trigger.text}" in: …{context}…')
    return "\n".join(lines)


def parse_verdicts(reply: str, count: int) -> list[bool] | None:
    """[is_instruction, ...] for each phrase, or None when the reply doesn't cover them all."""
    verdicts: dict[int, bool] = {}
    for match in _VERDICT_LINE.finditer(reply or ""):
        verdicts.setdefault(int(match.group(1)), match.group(2).lower() == "instruction")
    if any(n not in verdicts for n in range(1, count + 1)):
        return None
    return [verdicts[n] for n in range(1, count + 1)]


async def _ask_model(
    client: TextModel,
    model: str,
    timeout_s: float,
    transcript: str,
    triggers: list[Trigger],
    mode: str,
    limits: DailyLimits,
) -> list[bool] | None:
    """One model's verdicts, or None when it failed. Never raises for model trouble."""
    request = GeminiRequest(user_prompt=_judge_prompt(transcript, triggers))
    system_prompt = JUDGE_SYSTEM_PROMPT.format(target=_TARGETS[mode])
    try:
        reply = await asyncio.wait_for(
            client.generate_text(request, system_prompt, model, thinking_level="low"), timeout=timeout_s
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - any failure moves on to the next step
        if classify_error(exc) == GeminiErrorKind.DAILY_QUOTA:
            limits.mark_daily_limited(model)
        logger.warning("length check by %s failed: %s", model, exc)
        return None
    verdicts = parse_verdicts(reply, len(triggers))
    if verdicts is None:
        logger.warning("length check by %s: unreadable reply %r", model, reply[:200])
    return verdicts


# --- the whole check ----------------------------------------------------------------


@dataclass(frozen=True)
class LengthDecision:
    length: str | None = None
    note: str = ""  # footer (or Code-mode hint line) note; "" means nothing to say
    decided_by: str = "none"  # "none" (no trigger), "fast", "strong", "rules" or "default"


NOTE_TWO_LENGTHS = "heard two lengths, so normal length"
NOTE_NOT_CHECKED = "length words not checked, so normal length"


def _from_levels(levels: set[str], suffix: str, decided_by: str) -> LengthDecision:
    if not levels:
        return LengthDecision(None, "", decided_by)
    if len(levels) > 1:
        return LengthDecision(None, NOTE_TWO_LENGTHS, decided_by)
    level = next(iter(levels))
    return LengthDecision(level, f"length: {LABELS[level]}{suffix}", decided_by)


async def decide_length(
    transcript: str,
    mode: str,
    client: TextModel | None,
    fast_model: str,
    strong_model: str,
    limits: DailyLimits,
    fast_timeout_s: float = FAST_TIMEOUT_S,
    strong_timeout_s: float = STRONG_TIMEOUT_S,
) -> LengthDecision:
    """The length a voice question asks for (mode: "vision", "writing" or "code")."""
    triggers = find_triggers(transcript)
    if not triggers:
        return LengthDecision()

    if client is not None:
        for role, model, timeout_s in (("fast", fast_model, fast_timeout_s), ("strong", strong_model, strong_timeout_s)):
            if limits.is_daily_limited(model):
                continue
            verdicts = await _ask_model(client, model, timeout_s, transcript, triggers, mode, limits)
            if verdicts is not None:
                levels = {t.level for t, counts in zip(triggers, verdicts) if counts}
                return _from_levels(levels, "", role)

    verdicts = [rule_verdict(transcript, t) for t in triggers]
    counted = {t.level for t, v in zip(triggers, verdicts) if v == COUNTS}
    unsure = {t.level for t, v in zip(triggers, verdicts) if v == UNSURE}
    if not unsure:
        return _from_levels(counted, " (checked by simple rules)", "rules")
    if mode == "vision":
        return _from_levels(counted | unsure, " (not double-checked)", "default")
    if counted:
        return _from_levels(counted, " (checked by simple rules)", "rules")
    return LengthDecision(None, NOTE_NOT_CHECKED, "default")
