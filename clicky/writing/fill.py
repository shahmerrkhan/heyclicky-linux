"""Carrying out the check's plan (fields.py) under fixed, general safety rules.

The rules (they never depend on the kind of app):
  1. Text already in a box is never removed unless the user asked: a "replace"
     step must quote words that really are in the user's request. A box that
     has text gets nothing typed into it except by adding at its start or end.
     Text the model wants changed unasked is a stop (the reason is shown).
     The one exception: a follow-up may rewrite the text Clicky itself typed last
     time ("rewrite"), with no quote, but only when that text is found in the box
     exactly once; otherwise nothing is typed (Clicky can't tell which part is its own).
  2. Unsure: nothing is typed; the user clicks the box (the click wait below).
  3. Before each piece is typed, focus must be confirmed in that very box, and
     the box is read again (it must still be empty for a plain "type").
  4. Any failure: nothing more is typed; the caller shows and copies the rest.
Also: every email address, phone number or link typed must appear literally in
the user's words, the draft, or text already in a box, or nothing is typed.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from dataclasses import field as dc_field

from .blank import holds_only_objects, is_empty
from .fields import (
    ADD_END,
    ADD_START,
    REPLACE,
    REWRITE,
    TYPE,
    Earlier,
    Field,
    Place,
    Step,
    earlier_for,
    focused_field,
    move_to,
    place_label,
)
from .target import WritingTarget
from .typer import TypeResult, find_span

logger = logging.getLogger(__name__)

WAIT_S = 30.0  # the click wait and the name question
POLL_S = 0.4  # how often the click wait looks at which box has focus
QUOTE_MIN_WORDS = 2

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_LINK = re.compile(r"(?:https?://|www\.)[^\s<>\"'\])]+", re.IGNORECASE)
_PHONE = re.compile(r"\+?\d[\d \t().-]{5,}\d")
_DATE = re.compile(r"^\d{4}[-./]\d{1,2}[-./]\d{1,2}$|^\d{1,2}[-./]\d{1,2}[-./]\d{2,4}$")
_DONE = re.compile(r"^\W*done\W*$", re.IGNORECASE)

# Where text goes in the box, for the typer.
WHERE = {TYPE: "caret", ADD_START: "start", ADD_END: "end", REPLACE: "replace"}  # REWRITE: rewrite_part


def is_done(text: str) -> bool:
    """The user's "done" (typed or spoken: "Done.", "done!")."""
    return bool(_DONE.match(text or ""))


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _phones(text: str) -> list[str]:
    found = []
    for match in _PHONE.finditer(text):
        raw = match.group(0).strip()
        digits = re.sub(r"\D", "", raw)
        if len(digits) >= 7 and not _DATE.match(raw):
            found.append(digits)
    return found


def made_up_contacts(texts: list[str], sources: list[str]) -> list[str]:
    """Email addresses, phone numbers and links in `texts` that appear in none of `sources`."""
    pool = "\n".join(sources).lower()
    pool_phones = set()
    for source in sources:
        pool_phones.update(_phones(source))
    missing = []
    for text in texts:
        for email in _EMAIL.findall(text):
            if email.lower() not in pool:
                missing.append(email)
        for link in _LINK.findall(text):
            link = link.rstrip(".,;:!?")
            if link.lower() not in pool:
                missing.append(link)
        for digits in _phones(text):
            if not any(digits in p for p in pool_phones):
                missing.append(digits)
    return missing


def is_password_box(box: Field | None) -> bool:
    """A password box (accessibility role "password text"): Clicky never types a password."""
    return box is not None and "password" in (box.role or "").lower()


def quote_in_request(quote: str, request: str) -> bool:
    """A "replace" is allowed only when its quote is really in the user's words (whitespace and case aside)."""
    q = _squash(quote).strip(" \"'“”.,!?")
    return len(q.split()) >= QUOTE_MIN_WORDS and q in _squash(request)


def _not_empty_why(name: str, box: Field) -> str:
    """Why a plain "type" into this box isn't allowed (it isn't empty: blank.is_empty)."""
    if box.text is None:
        return f"Clicky can't read what's in {name}, so it won't type there"
    if holds_only_objects(box):
        return f"{name} already holds an image, attachment or other content"
    return f"{name} already has text in it"


def softened(step: Step, box: Field | None, earlier: list[Earlier] = ()) -> Step:
    """A plain "type" aimed at a multi-line box that has text becomes "add at the start".

    The model sometimes answers "type" for a message box that already holds something (a
    signature the app added, a quoted reply). Refusing leaves the user with nothing typed;
    adding before what is there removes nothing (rule 1). Not for a one-line box (a filled
    recipients or subject box is left alone), not for a box Clicky typed into last time (that
    is a rewrite), and not for a box that can't be read or holds only objects.
    """
    if step.do != TYPE or box is None or box.text is None or not box.multi_line:
        return step
    if is_empty(box) or holds_only_objects(box) or earlier_for(box, earlier) is not None:
        return step
    return replace(step, do=ADD_START)


def soften(steps: list[Step], earlier: list[Earlier] = ()) -> list[Step]:
    """`softened` for every step of a plan, before it is vetted."""
    out = [softened(s, s.place.field if s.place else None, earlier) for s in steps]
    changed = sum(1 for a, b in zip(steps, out, strict=True) if a is not b)
    if changed:
        logger.info("fill: %d step(s) changed from type to add at the start (the box has text that stays)", changed)
    return out


def vet(
    steps: list[Step], request: str, draft: str, fields: list[Field], earlier: list[Earlier] = ()
) -> str | None:
    """Why the plan can't be typed as it is (rule 1 and the contact check), or None when it can."""
    found = vet_rule(steps, request, draft, fields, earlier)
    return found[1] if found is not None else None


def vet_rule(
    steps: list[Step], request: str, draft: str, fields: list[Field], earlier: list[Earlier] = ()
) -> tuple[str, str] | None:
    """`vet` with the name of the rule that refused, for the log: (rule, why), or None."""
    for step in steps:
        box = step.place.field if step.place else None
        name = place_label(step.place) if step.place else (step.what or "the box")
        if step.do == REWRITE:
            mine = earlier_for(box, earlier)
            if mine is None or box is None or box.text is None or find_span(box.text, mine.text) is None:
                return "own_text_not_found", f"Clicky can't tell which part of {name} is the text it typed last time"
        if step.do == REPLACE and not quote_in_request(step.quote, request):
            return "replace_not_asked", (
                f"it would change the text already in {name}, and your request doesn't clearly ask for that")
        if step.do == TYPE and box is not None and not is_empty(box):
            rule = "box_unreadable" if box.text is None else (
                "box_has_objects" if holds_only_objects(box) else "box_not_empty")
            return rule, _not_empty_why(name, box)
    sources = [request, draft] + [f.text for f in fields if f.text]
    missing = made_up_contacts([s.text for s in steps], sources)
    if missing:
        return "made_up_contact", f"it would type {missing[0]}, which isn't in your words, the draft or the window"
    return None


def with_gap(step: Step, box: Field | None) -> str:
    """The text to add, kept apart from what is already in the box (a blank line, or a space on one line)."""
    existing = box.text if box is not None else None
    if step.do not in (ADD_START, ADD_END) or (box is not None and is_empty(box)):
        return step.text
    multi = box.multi_line if box is not None else True
    if not multi:
        return step.text + " " if step.do == ADD_START else " " + step.text
    if step.do == ADD_START:
        lead = existing or ""
        gap = "" if lead.startswith("\n\n") else "\n" if lead.startswith("\n") else "\n\n"
        return step.text + gap
    tail = existing or ""
    gap = "" if tail.endswith("\n\n") else "\n" if tail.endswith("\n") else "\n\n"
    return gap + step.text


def shown_text(steps: list[Step]) -> str:
    """For the bubble and the clipboard: one piece as it is; several as "Subject: …" lines, then the main text."""
    if len(steps) == 1:
        return steps[0].text
    lines, main = [], []
    for step in steps:
        if step.main:
            main.append(step.text)
            continue
        label = step.what or place_label(step.place)
        lines.append(f"{label[:1].upper()}{label[1:]}: {step.text}")
    return "\n".join(lines) + ("\n\n" + "\n\n".join(main) if main else "")


# --- typing the plan ---------------------------------------------------------------


@dataclass
class Typed:
    step: Step
    label: str  # the box it went in
    box: Field | None  # that box as it was just before typing
    result: TypeResult
    target: WritingTarget | None = None  # the click wait: the window of the box the user clicked


@dataclass
class FillOutcome:
    typed: list[Typed] = dc_field(default_factory=list)
    rest: list[Step] = dc_field(default_factory=list)  # not typed
    why: str = ""  # why it stopped before the end ("" when everything was typed)
    stopped: bool = False  # Esc
    code: str = ""  # the rule or failure that stopped it, for the log ("" when everything was typed)

    @property
    def complete(self) -> bool:
        return not self.rest and not self.why and not self.stopped


Move = Callable[[WritingTarget, Place], Awaitable[tuple[bool, str, Field | None]]]
TypeText = Callable[[WritingTarget, str, asyncio.Event, str], Awaitable[TypeResult]]  # DraftTyper.type_draft
Rewrite = Callable[[WritingTarget, str, str, asyncio.Event], Awaitable[TypeResult]]  # DraftTyper.rewrite_part


async def run_steps(
    target: WritingTarget,
    steps: list[Step],
    type_text: TypeText,
    cancel: asyncio.Event,
    activate: Callable[[WritingTarget], Awaitable[bool]],
    move: Move = move_to,
    earlier: list[Earlier] = (),
    rewrite: Rewrite | None = None,
) -> FillOutcome:
    """Type each piece into its box, in order. Stops at the first piece that can't be typed safely."""
    outcome = FillOutcome()
    for index, step in enumerate(steps):
        label = place_label(step.place)

        def stop(why: str, stopped: bool = False, index: int = index, code: str = "") -> FillOutcome:
            # A piece typed only partly stays in the rest, so the whole of it is shown and copied.
            outcome.rest, outcome.why, outcome.stopped = list(steps[index:]), why, stopped
            outcome.code = code or ("esc" if stopped else "")
            return outcome

        if cancel.is_set():
            return stop("", stopped=True)
        if step.place is None:
            return stop(f"no box was chosen for {step.what or 'the text'}", code="no_box")
        if not await activate(target):
            return stop(f"couldn't switch back to {target.app_name}", code="activate_failed")
        ok, why, box = await move(target, step.place)
        if not ok:
            return stop(f"couldn't reach {label} ({why})", code="focus_failed")
        label = place_label(step.place, box)
        if is_password_box(box):
            return stop(f"{label} is a password box, and Clicky never types into one", code="password_box")
        # The box as it is now: text that turned up since the check (a signature) stays, ours goes before it.
        step = softened(step, box, earlier)
        if step.do == TYPE and box is not None and not is_empty(box):
            if box.text is None:
                return stop(f"Clicky can't read what's in {label} now, so nothing was typed there",
                            code="box_unreadable")
            if holds_only_objects(box):
                return stop(f"{label} holds an image or attachment now, so nothing was typed there",
                            code="box_has_objects")
            return stop(f"{label} has text in it now, so nothing was typed there", code="box_not_empty")
        if cancel.is_set():
            return stop("", stopped=True)
        if step.do == REWRITE:
            mine = earlier_for(box or (step.place.field if step.place else None), earlier)
            if mine is None or rewrite is None:
                return stop(f"Clicky can't tell which part of {label} is the text it typed last time",
                            code="own_text_not_found")
            result = await rewrite(target, mine.text, step.text, cancel)
        else:
            result = await type_text(target, with_gap(step, box), cancel, WHERE[step.do])
        if result.stopped or not result.typed or result.why_not:
            if result.typed_chars:
                outcome.typed.append(Typed(step, label, box, result))
            if result.stopped:
                return stop("", stopped=True)
            return stop(f"{result.why_not or 'nothing could be typed'} (in {label})", code="typing_failed")
        outcome.typed.append(Typed(step, label, box, result))
    return outcome


def _join(labels: list[str]) -> str:
    if len(labels) <= 1:
        return "".join(labels)
    return ", ".join(labels[:-1]) + " and " + labels[-1]


def outcome_line(outcome: FillOutcome, focused_before: Field | None, copied: bool) -> str:
    """The explanation strip: what went where, in plain English."""
    full = [t.label for t in outcome.typed if not _partly(t)]
    copy = " It's copied to your clipboard." if copied else ""
    rewrote = [t.label for t in outcome.typed if t.step.do == REWRITE and not _partly(t)]
    if rewrote:
        copy = f" In {_join(rewrote)}, only Clicky's earlier text was replaced; the rest was left alone.{copy}"
    if outcome.complete:
        line = f"Typed into {_join(full)}."
        if rewrote:
            return line + copy
        if focused_before is not None and not any(t.box is not None and t.box.same_box(focused_before)
                                                   for t in outcome.typed):
            line = line[:-1] + f" (focus was in {focused_before.label})."
        return line
    done = f"Typed into {_join(full)}. " if full else ""
    done += "".join(f"Typed {t.result.typed_chars} characters into {t.label}. " for t in outcome.typed if _partly(t))
    done = done or "Nothing was typed. "
    if outcome.stopped:
        return f"{done}Stopped; the rest is above, in full.{copy}"
    return f"{done}Stopped: {outcome.why}. The rest is above, in full.{copy}"


def _partly(t: Typed) -> bool:
    return t.result.stopped or bool(t.result.why_not)


def remember_typed(earlier: list[Earlier], typed: list[Typed]) -> list[Earlier]:
    """What Clicky has typed in the window after this run, for the next follow-up.

    Pieces typed in full are remembered with their box; an earlier piece whose box got
    something this time is dropped (replaced, or no longer exactly as typed).
    """
    touched = [t.box for t in typed if t.box is not None]
    kept = [e for e in earlier if e.box is None or not any(e.box.same_box(b) for b in touched)]
    new = [Earlier(t.step.text, t.step.what, t.label, t.box) for t in typed if not _partly(t) and t.box is not None]
    return kept + new


# --- the click wait ----------------------------------------------------------------


@dataclass
class WaitOutcome:
    typed: list[Typed] = dc_field(default_factory=list)
    rest: list[Step] = dc_field(default_factory=list)
    ended: str = ""  # "all", "done", "timeout", "message" or "failed"
    message: str = ""  # "message": what the user sent instead (it runs as a new question)
    why: str = ""  # "failed": why the typing didn't work


Poll = Callable[[], Awaitable[tuple[WritingTarget, Field] | None]]
Match = Callable[[list[Step], Field, WritingTarget], Awaitable[tuple[int | None, str]]]
TypeInto = Callable[[WritingTarget, Field, str], Awaitable[TypeResult]]


async def click_wait(
    pieces: list[Step],
    poll: Poll,
    replies: asyncio.Queue,
    match: Match,
    type_into: TypeInto,
    on_status: Callable[[str], None],
    wait_s: float = WAIT_S,
    poll_s: float = POLL_S,
) -> WaitOutcome:
    """Wait for the user to click the box(es); nothing is typed until they do.

    One piece: it goes in the box that had focus last when the user replies "done".
    Several pieces: each box the user clicks gets the piece that fits it (no "done"
    needed in between); the wait ends once all are typed, or early on "done".
    Anything else the user sends ends the wait (it runs as a new question). After
    `wait_s` without a piece typed, the wait ends with nothing more typed.
    """
    rest, outcome = list(pieces), WaitOutcome()
    first = await poll()
    baseline = first[1] if first else None  # focus at the start isn't a click
    last: tuple[WritingTarget, Field] | None = first
    handled: Field | None = baseline
    deadline = time.monotonic() + wait_s
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            outcome.ended = "timeout"
            break
        try:
            reply = await asyncio.wait_for(replies.get(), timeout=min(poll_s, remaining))
        except TimeoutError:
            reply = None
        if reply is not None:
            if not is_done(reply):
                outcome.ended, outcome.message = "message", reply
                break
            if len(pieces) > 1:
                outcome.ended = "done"
                break
            if last is None:
                on_status("Click the box first, then reply \"done\".")
                continue
            result = await type_into(last[0], last[1], rest[0].text)
            if not result.typed or result.why_not or result.stopped:
                if result.typed_chars:
                    outcome.typed.append(Typed(rest[0], last[1].label, last[1], result, last[0]))
                outcome.ended, outcome.why = "failed", result.why_not or "stopped"
                break
            outcome.typed.append(Typed(rest[0], last[1].label, last[1], result, last[0]))
            rest = []
            outcome.ended = "all"
            break
        now = await poll()
        if now is None:
            continue
        last = now
        if len(pieces) == 1 or (handled is not None and now[1].same_box(handled)):
            continue
        handled = now[1]
        index, why = await match(rest, now[1], now[0])
        if index is None:
            on_status(f"Nothing typed in {now[1].label}: {why}. Still waiting; click another box.")
            continue
        step = rest[index]
        result = await type_into(now[0], now[1], step.text)
        if not result.typed or result.why_not or result.stopped:
            if result.typed_chars:
                outcome.typed.append(Typed(step, now[1].label, now[1], result, now[0]))
            outcome.ended, outcome.why = "failed", result.why_not or "stopped"
            break
        outcome.typed.append(Typed(step, now[1].label, now[1], result, now[0]))
        rest.pop(index)
        if not rest:
            outcome.ended = "all"
            break
        deadline = time.monotonic() + wait_s  # time for the next box
        on_status(f"Typed into {now[1].label}. Now click the box for: {_join([s.what or 'the text' for s in rest])}.")
    outcome.rest = rest
    return outcome


async def type_into_clicked(
    target: WritingTarget,
    box: Field,
    text: str,
    activate: Callable[[WritingTarget], Awaitable[bool]],
    type_text: TypeText,
    cancel: asyncio.Event,
) -> TypeResult:
    """The click wait: type into the box the user clicked, at their caret (nothing is removed).

    Focus must still (or again, after "done" in the input box) be in that very box.
    """
    if not await activate(target):
        return TypeResult(False, why_not=f"couldn't switch back to {target.app_name}")
    now = await focused_field(target.pid) if target.pid is not None else None
    if now is None or not now.same_box(box):
        return TypeResult(False, why_not="focus isn't in the box you clicked any more")
    if is_password_box(now):
        return TypeResult(False, why_not="that's a password box, and Clicky never types into one")
    return await type_text(target, text, cancel, "caret")
