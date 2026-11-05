"""What a Write draft becomes, and where each piece goes, in the window the user was in.

Before any draft is typed (every Write draft, however it started; never for terminals):
  1. a screenshot of the whole screen (all monitors), Clicky's own windows hidden;
  2. the window's text boxes, from accessibility (name, kind, position, focus, the text in it);
  3. a model works out from the screen what kind of app and box this is and writes
     the final text for each box that should get some, in typing order:
     the fast model first (FAST_TIMEOUT_S), the strong model as backup (STRONG_TIMEOUT_S);
     a model remembered as daily-limited is skipped.

The result is one of:
  - "go": a box for every piece. A box from the list is reached through
    accessibility (focus moved, then confirmed). A box the model can see but
    the list misses is reached with a click on the point it gives (the backup),
    and the pointer is put back where it was.
  - "unsure": the model can't tell which box is meant. Nothing is typed; the
    user is asked to click the box.
  - "stop": text already in a box would have to change and the user didn't ask
    for that, or the task can't be done there. Nothing is typed; the reason is shown.
  - "failed": no screenshot, no field list, both models failed or were too slow,
    or an unreadable reply. Nothing is typed.

A follow-up in the same window ("make it shorter", "also write the subject") goes
through the same check: the model also gets what Clicky typed last time and into
which boxes ("earlier"). It may rewrite that text ("rewrite": only the part Clicky
typed is replaced; Clicky finds it in the box, and the rest of the box stays) and
fill boxes that are still empty.

The general safety rules (nothing removed unasked, made-up addresses, focus
confirmed before typing) are applied to the result in fill.py.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from dataclasses import field as dc_field

from ..llm.gemini import GeminiErrorKind, GeminiRequest, LabeledImage, classify_error
from ..llm.length import DailyLimits, TextModel
from ..screen.capture import CapturedScreen
from .target import WritingTarget, active_window_target
from .typer import _run, atspi

logger = logging.getLogger(__name__)

FAST_TIMEOUT_S = 4.0
STRONG_TIMEOUT_S = 6.0
MATCH_TIMEOUT_S = 4.0  # the click wait: which piece fits the box the user clicked
NORMALIZED_MAX = 1000

GO, UNSURE, STOP, FAILED = "go", "unsure", "stop", "failed"
TYPE, ADD_START, ADD_END, REPLACE = "type", "add_start", "add_end", "replace"
REWRITE = "rewrite"  # a follow-up: the text Clicky typed last time in that box, swapped for a new version
ACTIONS = (TYPE, ADD_START, ADD_END, REPLACE, REWRITE)
NAME_SLOT = "{MY_NAME}"  # where the user's own name goes (filled in by Clicky)

# The bubble text when nothing could be typed (the draft follows it).
FAILED_TEXT = "Couldn't find a suitable place to type or paste the text in. Here is what you can type in it:"

PLAN_SYSTEM_PROMPT = """\
you decide what clicky types, and where. the user asked clicky to write something. you get their request, clicky's draft, screenshots of the user's whole screen, and a numbered list of the text boxes in the window the text is for (from accessibility: name, kind, where it is on the screenshots, whether it has focus, and the text already in it).

work out from the screen what kind of app and box this is. then write the final text for each box that should get some, and leave every other box alone.

what to write:
- base it on the draft and the request: keep every fact, the language and the length of the draft's message. never invent facts.
- an email app (gmail, outlook or similar, a compose or reply window): a full email. the recipient's address goes in the recipients box, the subject in the subject box, and the body in the message box, written as:
  Hey <their name>,
  <blank line>
  <the message>
  <blank line>
  Thanks,  (for a request or a favour; otherwise Best regards,)
  {MY_NAME}
  write "Hey," when you don't know the recipient's name; if the user names the person, use that name.
  if the message box already holds a signature the app adds by itself, leave out the closing line and {MY_NAME}.
  a recipients or subject box that already has text (a reply) is left alone.
- a notepad or text editor: the message only, no greeting or sign-off unless the user asked for one.
- a messaging or chat app: a plain message, no subject, no sign-off.
- any other form: each box gets the value that belongs in it.
- if the window has an empty box where something clearly belongs (like an email subject), write it from the message, even when the user didn't give it. that isn't inventing facts. (addresses, phone numbers and links are the exception: see below.)
- write {MY_NAME} wherever the user's own name goes; never write their name yourself.
- email addresses, phone numbers and links: only ones that appear in the user's request, the draft, or already in a box. never make one up; if one is needed and you don't have it, leave that box alone.

how each box gets its text ("do"):
- "type": the box is empty.
- "add_start" or "add_end": the box has text that must stay (a signature, earlier lines); your text goes before or after it. nothing is removed.
- "replace": removes the box's text and puts yours in. only when the user's request clearly asks to change that text, and then "quote" must be the exact words of the request that ask for it.
- "rewrite": only for a box clicky typed into last time (listed under "what clicky typed last time"). your text replaces just the part clicky typed there (clicky finds it); everything else in the box (a signature, a quoted reply, anything else) stays. no quote needed.
if text already in a box would have to change and the user didn't ask for that, use status "stop".

follow-ups: when the prompt says what clicky typed last time, the request may build on it.
- the request changes that text ("make it shorter", "more formal", "add that I'm free on monday"): "rewrite" it in its box, keeping the format it had (greeting, sign-off).
- the request asks for something more ("also write the subject"): fill that box ("type" if it's empty), and leave clicky's other text alone unless the request changes it too.
- the draft may hold several pieces at once (a "Subject: ..." line above a message): put each piece in its own box. never put a label like "Subject:" into the message.

where each piece goes:
- "box": the number from the list.
- if the right box is clearly visible on a screenshot but missing from the list, give "point": [x, y] (its centre, 0 to 1000 across and down that screen's image) and "screen" (the number in the image label) instead.
- the steps are typed in the order you list them.
- mark the step that holds the main text (the message or the body) with "main": true.
- "what": a few words for what the text is ("recipient address", "subject", "message").

status:
- "go": you know where everything goes.
- "unsure": you can't tell which box is meant. still give the steps, without "box" or "point".
- "stop": existing text would have to change unasked, or this can't be done here. "why": one short sentence for the user.

"name_on_screen": the user's own name if the screen clearly shows it (the signed-in account, a "from" line), else null.

reply with json only, like:
{"status": "go", "why": "", "name_on_screen": null, "steps": [{"box": 2, "do": "type", "what": "recipient address", "text": "sam@example.com"}, {"box": 4, "do": "add_start", "what": "message", "main": true, "text": "Hey Sam,\\n\\nI'll be there by 5.\\n\\nBest regards,\\n{MY_NAME}"}], "leave": [{"box": 1, "why": "already has text"}]}
"""

MATCH_SYSTEM_PROMPT = """\
the user just clicked a text box so clicky can type into it. you get that box (from accessibility) and a numbered list of the pieces of text still to be typed, each with what it is.
pick the piece that belongs in that box. if none belongs there, say why in one short sentence for the user.
reply with json only, exactly like {"piece": 2} or {"piece": null, "why": "that's the search box"}.
"""


@dataclass(frozen=True)
class Field:
    id: str
    role: str
    name: str
    description: str
    multi_line: bool
    focused: bool
    x: int
    y: int
    w: int
    h: int
    readable: bool
    text: str | None
    length: int

    @property
    def centre(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2

    def contains(self, gx: int, gy: int) -> bool:
        return self.w > 0 and self.h > 0 and self.x <= gx < self.x + self.w and self.y <= gy < self.y + self.h

    def same_box(self, other: Field) -> bool:
        """The same box on screen (ids differ between `fields` and `focused`; a rich-text
        box may report the focused part inside it, so either one's centre inside the other counts)."""
        if self.w > 0 and other.w > 0:
            return self.contains(*other.centre) or other.contains(*self.centre)
        return (self.name, self.role) == (other.name, other.role)

    @property
    def label(self) -> str:
        """For the bubble: "Message Body", "To recipients"."""
        return self.name or self.description or self.role or "the text box"


@dataclass(frozen=True)
class Place:
    field: Field | None = None  # reached through accessibility
    point: tuple[int, int] | None = None  # screen pixels: the backup click


@dataclass(frozen=True)
class Step:
    """One piece of text and the box it goes in."""

    text: str
    what: str = ""  # what the text is ("subject"), for the bubble and the click wait
    do: str = TYPE
    place: Place | None = None  # None: no box chosen (unsure)
    main: bool = False  # the main text: a follow-up ("make it shorter") may replace it
    quote: str = ""  # "replace": the words of the request that ask for it


@dataclass(frozen=True)
class Earlier:
    """What Clicky typed into one box last time (a follow-up may rewrite it)."""

    text: str
    what: str = ""
    label: str = ""  # the box, for the bubble
    box: Field | None = None  # that box as it was just before typing (matched by position)


def earlier_for(box: Field | None, earlier: list[Earlier]) -> Earlier | None:
    """What Clicky typed last time into this box, if anything."""
    if box is None:
        return None
    return next((e for e in earlier if e.box is not None and e.box.same_box(box)), None)


@dataclass
class CheckResult:
    status: str  # GO, UNSURE, STOP or FAILED
    steps: list[Step] = dc_field(default_factory=list)
    why: str = ""  # STOP / FAILED / UNSURE: the reason, for the log and the bubble
    name_on_screen: str | None = None
    decided_by: str = ""  # "fast" or "strong"
    fields: list[Field] = dc_field(default_factory=list)
    focused: Field | None = None  # where focus was when the check ran
    reply: dict = dc_field(default_factory=dict)  # the model's whole reply
    images: list[LabeledImage] = dc_field(default_factory=list)  # the screenshot, for the click wait


Capture = Callable[[], Awaitable[list[CapturedScreen]]]
ListFields = Callable[[int], Awaitable[list[Field] | None]]


def _field_from(f: dict) -> Field:
    return Field(
        id=str(f.get("id", "")), role=str(f.get("role", "")), name=str(f.get("name", "")),
        description=str(f.get("description", "")), multi_line=bool(f.get("multi_line")),
        focused=bool(f.get("focused")), x=int(f.get("x", -1)), y=int(f.get("y", -1)),
        w=int(f.get("w", -1)), h=int(f.get("h", -1)), readable=bool(f.get("readable")),
        text=f.get("text"), length=int(f.get("length", 0)),
    )


async def list_fields(pid: int) -> list[Field] | None:
    """The editable fields in the app's active window (None when accessibility can't list them)."""
    result = await atspi("fields", pid)
    if not result.get("ok"):
        logger.info("field list not available: %s", result.get("why"))
        return None
    fields = []
    for f in result.get("fields", []):
        try:
            fields.append(_field_from(f))
        except (TypeError, ValueError):
            continue
    return fields


async def focused_field(pid: int) -> Field | None:
    """The editable box that has focus in that app right now (None: none, or it can't be told)."""
    result = await atspi("focused", pid)
    if not result.get("ok") or not isinstance(result.get("field"), dict):
        return None
    try:
        return _field_from(result["field"])
    except (TypeError, ValueError):
        return None


def _normalized(screens: list[CapturedScreen], gx: int, gy: int) -> str:
    """Where a screen pixel is on the screenshots, as the model sees them: "screen1 at (512, 220)"."""
    for s in screens:
        m = s.monitor
        if m.contains_global_point(gx, gy):
            nx = round((gx - m.x) / max(1, m.width - 1) * NORMALIZED_MAX)
            ny = round((gy - m.y) / max(1, m.height - 1) * NORMALIZED_MAX)
            return f"screen{m.index} at ({nx}, {ny})"
    return "off screen"


def _box_text(f: Field) -> str:
    if f.text is None:
        return "can't be read"
    if not f.text.strip():
        return "empty"
    shown = f.text if len(f.text) <= 600 else f.text[:300] + " … " + f.text[-300:]
    return json.dumps(shown, ensure_ascii=False)


def _describe_field(number: int, f: Field, screens: list[CapturedScreen], earlier: list[Earlier]) -> str:
    kind = "multi-line" if f.multi_line else "one line"
    where = _normalized(screens, *f.centre) if f.w > 0 else "position unknown"
    focus = ", HAS FOCUS" if f.focused else ""
    mine = ", clicky typed into it last time" if earlier_for(f, earlier) is not None else ""
    return f'{number}. "{f.label}" ({f.role}, {kind}{focus}{mine}), centre {where}, text: {_box_text(f)}'


def _prompt(
    draft: str,
    request: str,
    window: WritingTarget,
    fields: list[Field],
    screens: list[CapturedScreen],
    earlier: list[Earlier] = (),
) -> str:
    lines = [f'window: {window.app_class} — "{window.title}"', "", "text boxes in that window:"]
    lines += [_describe_field(i, f, screens, earlier) for i, f in enumerate(fields, start=1)] or ["(none listed)"]
    if earlier:
        lines += ["", "what clicky typed last time in this window (the user may have changed it since; the box list shows what each box holds now):"]
        lines += [
            f'- into "{e.label}" ({e.what or "text"}): {json.dumps(e.text, ensure_ascii=False)}' for e in earlier
        ]
    lines += [
        "",
        f"what the user asked for: {json.dumps(request, ensure_ascii=False)}",
        "",
        f"clicky's draft: {json.dumps(draft, ensure_ascii=False)}",
    ]
    return "\n".join(lines)


def fields_summary(fields: list[Field], app_class: str) -> str:
    """For the log: the boxes' names, kinds and whether each is empty — never their text."""
    if not fields:
        return f"no text boxes listed in {app_class}"
    parts = []
    for i, f in enumerate(fields, start=1):
        filled = "unreadable" if f.text is None else "empty" if not f.text.strip() else "has text"
        focus = ", focused" if f.focused else ""
        parts.append(f'{i} "{f.label}" ({f.role}, {"multi-line" if f.multi_line else "one line"}, {filled}{focus})')
    return f"{len(fields)} text boxes in {app_class}: " + "; ".join(parts)


def plan_summary(result: CheckResult) -> str:
    """For the log: the plan as actions and box numbers — never the text."""
    numbers = {f.id: i for i, f in enumerate(result.fields, start=1)}
    steps = []
    for s in result.steps:
        if s.place is None:
            where = "no box"
        elif s.place.field is not None:
            where = f"box {numbers.get(s.place.field.id, '?')}"
        else:
            where = "a point"
        steps.append(f"{s.do} {where}" + (" (main)" if s.main else ""))
    return f"plan by {result.decided_by or '?'}: {result.status}" + (": " + ", ".join(steps) if steps else "")


def _json_object(reply: str) -> dict | None:
    start, end = reply.find("{"), reply.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(reply[start : end + 1])
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _place(value: dict, fields: list[Field], screens: list[CapturedScreen]) -> Place | None:
    """A step's box or point as a place (None: unreadable)."""
    if "box" in value and value["box"] is not None:
        try:
            number = int(value["box"])
        except (TypeError, ValueError):
            return None
        return Place(field=fields[number - 1]) if 1 <= number <= len(fields) else None
    if "point" in value and value["point"] is not None:
        try:
            nx, ny = (int(v) for v in value["point"])
            screen_number = int(value.get("screen") or 0) or None
        except (TypeError, ValueError):
            return None
        if not (0 <= nx <= NORMALIZED_MAX and 0 <= ny <= NORMALIZED_MAX):
            return None
        screen = next((s for s in screens if s.monitor.index == screen_number), None) or next(
            (s for s in screens if s.is_primary_focus), screens[0] if screens else None
        )
        if screen is None:
            return None
        gx, gy = screen.global_point_from_normalized(nx, ny)
        # A point on a listed field: reach that field through accessibility instead of clicking.
        hit = next((f for f in fields if f.contains(gx, gy)), None)
        return Place(field=hit) if hit is not None else Place(point=(gx, gy))
    return None


def read_reply(reply: str, fields: list[Field], screens: list[CapturedScreen]) -> CheckResult | None:
    """The model's answer as a result, or None when it's unreadable (the next model is asked)."""
    data = _json_object(reply)
    if data is None:
        return None
    status = data.get("status")
    if status not in (GO, UNSURE, STOP):
        return None
    why = data.get("why") if isinstance(data.get("why"), str) else ""
    name = data.get("name_on_screen")
    name = name.strip() if isinstance(name, str) and name.strip() else None
    raw = data.get("steps", [])
    if not isinstance(raw, list):
        return None
    steps: list[Step] = []
    for value in raw:
        if not isinstance(value, dict):
            return None
        text, do = value.get("text"), value.get("do", TYPE)
        if not isinstance(text, str) or not text.strip() or do not in ACTIONS:
            return None
        place = None
        if status == GO:
            place = _place(value, fields, screens)
            if place is None:
                return None
        quote = value.get("quote") if isinstance(value.get("quote"), str) else ""
        what = value.get("what") if isinstance(value.get("what"), str) else ""
        steps.append(Step(text.strip("\n"), what.strip(), do, place, value.get("main") is True, quote))
    if status == GO:
        if not steps:
            return None
        boxes = [s.place.field.id for s in steps if s.place and s.place.field] + [
            str(s.place.point) for s in steps if s.place and s.place.point
        ]
        if len(boxes) != len(set(boxes)):
            return None  # two pieces for one box
    # One main text: the first one marked, or the only step.
    mains = [i for i, s in enumerate(steps) if s.main]
    main = mains[0] if mains else (0 if len(steps) == 1 else None)
    steps = [Step(s.text, s.what, s.do, s.place, i == main, s.quote) for i, s in enumerate(steps)]
    return CheckResult(status, steps=steps, why=why.strip(), name_on_screen=name, reply=data)


async def check_plan(
    draft: str,
    request: str,
    target: WritingTarget | None,
    client: TextModel | None,
    fast_model: str,
    strong_model: str,
    limits: DailyLimits,
    capture: Capture,
    fields_of: ListFields = list_fields,
    fast_timeout_s: float = FAST_TIMEOUT_S,
    strong_timeout_s: float = STRONG_TIMEOUT_S,
    earlier: list[Earlier] = (),
) -> CheckResult:
    """What to type into which box of the target window, for this draft (`earlier`: a follow-up)."""
    if target is None or target.pid is None:
        return CheckResult(FAILED, why="there was no window to type into")
    if client is None:
        return CheckResult(FAILED, why="Gemini isn't set up")
    try:
        screens = await capture()
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - no screenshot: nothing is typed
        logger.warning("where-to-type screenshot failed: %s", exc)
        screens = []
    if not screens:
        return CheckResult(FAILED, why="couldn't take a screenshot")
    fields = await fields_of(target.pid)
    if fields is None:
        return CheckResult(FAILED, why=f"couldn't list the text boxes in {target.app_name}")
    logger.info("where-to-type: %s", fields_summary(fields, target.app_class))
    if earlier:
        logger.info("where-to-type: a follow-up, Clicky typed into %d box(es) last time", len(earlier))
    focused = next((f for f in fields if f.focused), None)
    images = [LabeledImage(data=s.image_data, label=s.label_for_model()) for s in screens]
    request_obj = GeminiRequest(user_prompt=_prompt(draft, request, target, fields, screens, earlier), images=images)

    for role, model, timeout_s in (("fast", fast_model, fast_timeout_s), ("strong", strong_model, strong_timeout_s)):
        if limits.is_daily_limited(model):
            continue
        try:
            reply = await asyncio.wait_for(
                client.generate_text(request_obj, PLAN_SYSTEM_PROMPT, model, thinking_level="low"), timeout=timeout_s
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - the next model, then give up
            if classify_error(exc) == GeminiErrorKind.DAILY_QUOTA:
                limits.mark_daily_limited(model)
            logger.warning("where-to-type check by %s failed: %s", model, exc)
            continue
        result = read_reply(reply, fields, screens)
        if result is None:
            logger.warning(
                "where-to-type check by %s: unreadable reply (%d characters)", model, len(reply or "")
            )
            continue
        if result.status == UNSURE and not result.steps:
            result.steps = [Step(draft, "the text", main=True)]
        if result.status == UNSURE and not result.why:
            result.why = "it couldn't tell which box is meant"
        result.decided_by, result.fields, result.focused, result.images = role, fields, focused, images
        logger.info("where-to-type: %s", plan_summary(result))
        return result
    return CheckResult(FAILED, why="couldn't work out where the text goes", fields=fields, focused=focused, images=images)


async def match_piece(
    pieces: list[Step],
    box: Field,
    window: WritingTarget,
    client: TextModel | None,
    fast_model: str,
    limits: DailyLimits,
    images: list[LabeledImage],
    timeout_s: float = MATCH_TIMEOUT_S,
) -> tuple[int | None, str]:
    """The click wait: which of `pieces` belongs in the box the user clicked (index, or None and why).

    One piece needs no call: it goes in whatever box is clicked.
    """
    if len(pieces) == 1:
        return 0, ""
    unknown = None, "Clicky couldn't work out which piece goes in that box"
    if client is None or limits.is_daily_limited(fast_model):
        return unknown
    kind = "multi-line" if box.multi_line else "one line"
    lines = [
        f'window: {window.app_class} — "{window.title}"',
        f'the box the user clicked: "{box.label}" ({box.role}, {kind}), text: {_box_text(box)}',
        "",
        "pieces still to type:",
    ]
    for i, piece in enumerate(pieces, start=1):
        preview = piece.text if len(piece.text) <= 300 else piece.text[:300] + " …"
        lines.append(f"{i}. {piece.what or 'text'}: {json.dumps(preview, ensure_ascii=False)}")
    request = GeminiRequest(user_prompt="\n".join(lines), images=list(images))
    try:
        reply = await asyncio.wait_for(
            client.generate_text(request, MATCH_SYSTEM_PROMPT, fast_model, thinking_level="low"), timeout=timeout_s
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - say so and keep waiting
        if classify_error(exc) == GeminiErrorKind.DAILY_QUOTA:
            limits.mark_daily_limited(fast_model)
        logger.warning("piece match failed: %s", exc)
        return unknown
    data = _json_object(reply or "")
    if data is None or "piece" not in data:
        return unknown
    if data["piece"] is None:
        why = data.get("why") if isinstance(data.get("why"), str) and data.get("why").strip() else ""
        return None, why.strip() or "none of the pieces belongs in that box"
    try:
        number = int(data["piece"])
    except (TypeError, ValueError):
        return unknown
    return (number - 1, "") if 1 <= number <= len(pieces) else unknown


# --- reaching the box -----------------------------------------------------------------


async def pointer_position() -> tuple[int, int] | None:
    try:
        _, out = await _run("xdotool", "getmouselocation", "--shell", timeout=3)
    except (TimeoutError, OSError):
        return None
    values = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    try:
        return int(values["X"]), int(values["Y"])
    except (KeyError, ValueError):
        return None


def _is_active(target: WritingTarget) -> bool:
    active = active_window_target()
    return active is not None and active.window_id == target.window_id


async def click_at(target: WritingTarget, point: tuple[int, int]) -> tuple[bool, str, Field | None]:
    """The backup: click the point in the remembered window, then put the pointer back.

    Only while that window is active; afterwards it must still be active and the
    editable box under the point must have focus, or it counts as failed.
    """
    if not _is_active(target):
        return False, f"{target.app_name} isn't the active window", None
    if not (target.x <= point[0] < target.x + target.width and target.y <= point[1] < target.y + target.height):
        return False, "that spot isn't inside the window", None
    before = await pointer_position()
    try:
        await _run("xdotool", "mousemove", "--sync", str(point[0]), str(point[1]), "click", "1", timeout=3)
    except (TimeoutError, OSError):
        return False, "couldn't click there", None
    finally:
        if before is not None:
            try:
                await _run("xdotool", "mousemove", "--sync", str(before[0]), str(before[1]), timeout=3)
            except (TimeoutError, OSError):
                logger.warning("couldn't put the pointer back")
    await asyncio.sleep(0.2)
    if not _is_active(target):
        return False, "the click moved away from that window", None
    box = await focused_field(target.pid) if target.pid is not None else None
    if box is None:
        return False, "focus didn't move to a text box", None
    if box.w > 0 and not box.contains(*point):
        return False, "focus went to a different box than the one clicked", None
    return True, "", box


async def move_to(target: WritingTarget, place: Place) -> tuple[bool, str, Field | None]:
    """Move focus to the place (accessibility for a listed field, else the backup click) and confirm it.

    Returns the box that has focus afterwards, read again, so the caller sees its text as it is now.
    """
    if place.field is not None:
        if target.pid is None:
            return False, "the window has no process", None
        result = await atspi("focus", target.pid, place.field.id)
        if not result.get("ok"):
            return False, str(result.get("why", "")), None
        box = await focused_field(target.pid)
        if box is None or not box.same_box(place.field):
            return False, "focus didn't stay in that box", None
        return True, "", box
    if place.point is not None:
        return await click_at(target, place.point)
    return False, "no place to move to", None


def place_label(place: Place | None, box: Field | None = None) -> str:
    """For the bubble: where a piece went."""
    if place is not None and place.field is not None:
        return place.field.label
    if box is not None:
        return box.label
    return "the box Clicky clicked"
