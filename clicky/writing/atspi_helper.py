"""AT-SPI helper for writing mode. Runs as a separate program with the system
Python (it needs the GObject bindings, which Clicky's own environment doesn't
have): `/usr/bin/python3 -I atspi_helper.py <command> <pid>`, text on stdin.

Commands (each prints one JSON line):
  insert <pid> [start|end]
                 insert stdin text at the caret of the focused editable field
                 (with "start" or "end": at the very start or end of the field,
                 next to anything already there) → {"ok": true, "offset": <offset>, "length": <chars>}
  read <pid>     the focused field's whole text (for safe draft replacement),
                 and whether it can be edited directly (native apps) or only
                 read (Chrome, Opera) → {"ok": true, "text": "...", "editable": bool}.
                 Text that comes back with object placeholders (U+FFFC) can't
                 be read as plain text, so it counts as unreadable.
  replace <pid> <offset> <length>
                 delete <length> chars at <offset>, then insert stdin text there
                 → {"ok": true}
  select <pid> <start> <end>
                 select that part of the focused field's text, then read the
                 selection back → {"ok": true, "text": "<the selected text>"}
  fields <pid>   the editable fields showing in the app's active window:
                 → {"ok": true, "window": "<title>", "fields": [{"id", "role", "name",
                 "description", "multi_line", "focused", "x", "y", "w", "h",
                 "readable", "text", "length"}, ...]}; "text" is the field's text
                 (long text: its start and end), null when it can't be read.
                 A box with no visible text and a count above 0 also gets "placeholders",
                 "blocks_empty", "objects" (roles) and "codes" (character codes, never text).
                 A field inside another editable field (the paragraphs of a
                 rich-text box) is left out: the outer box is the field.
  focus <pid> <id>
                 move focus to that field (an id from `fields`) and confirm it
                 → {"ok": true}
  focused <pid>  the editable field that has focus (after a click), described
                 like one entry of `fields` (its "id" is empty)
                 → {"ok": true, "field": {...}}
  buttons <pid>  the buttons (links, menu items, tabs, …) showing in the app's active window,
                 at most BUTTON_LIMIT, in reading order → {"ok": true, "buttons": [{"id", "role",
                 "name", "description", "extra", "x", "y", "w", "h"}, ...]}; "extra" is other
                 accessibility text (attributes), for Clicky's forbidden-word check only
  press <pid> <id> <name>
                 press that button (an id from `buttons`) through its accessibility action,
                 only if it is still there, still has exactly that name and isn't a text box
                 → {"ok": true}
  at <pid> <x> <y>
                 what is at that screen point in the app: {"ok": true, "found": false} or
                 {"ok": true, "found": true, "kind": "text" | "control" | "other", "role",
                 "name", "description", "extra"}; "text" is an editable box, "control" a
                 button, link or anything else that can be pressed (or a part of one)

Failures print {"ok": false, "why": "..."} and exit 0, so Clicky can move on
to the next method.
"""

import json
import sys
import time

WALK_LIMIT_S = 5.0  # the fallback walk only; the built-in search is much faster
FIELD_LIMIT = 60  # at most this many fields are listed
TEXT_HEAD, TEXT_TAIL = 1200, 800  # a long field's text: its start and its end (signatures sit at the end)
FOCUS_WAIT_S = 1.0
BUTTON_LIMIT = 60  # at most this many buttons are listed
BUTTON_SCAN_LIMIT = 400  # objects looked at for them
BUTTON_WALK_LIMIT_S = 2.0
# Role names (get_role_name) of things that can be pressed.
PRESSABLE_ROLES = {
    "push button", "toggle button", "split button", "button", "link", "menu item", "check menu item",
    "radio menu item", "menu", "page tab", "check box", "radio button", "combo box", "list item", "tree item",
    "icon", "image map", "switch",
}
PRESS_ACTIONS = ("press", "click", "activate", "jump", "open", "toggle", "select")
PLACEHOLDER = "\ufffc"  # stands in for a child object (e.g. one line of a rich-text box)


def _search(root, Atspi):
    """The focused editable field under `root` via the app's built-in search, or None.

    Chrome, Opera and GTK apps answer this without Clicky walking every object.
    """
    try:
        states = Atspi.StateSet.new([Atspi.StateType.FOCUSED, Atspi.StateType.EDITABLE])
        rule = Atspi.MatchRule.new(
            states, Atspi.CollectionMatchType.ALL,
            {}, Atspi.CollectionMatchType.ALL,
            [], Atspi.CollectionMatchType.ALL,
            [], Atspi.CollectionMatchType.ALL,
            False,
        )
        matches = Atspi.Collection.get_matches(root, rule, Atspi.CollectionSortOrder.CANONICAL, 1, True)
    except Exception:  # noqa: BLE001 - not supported here: fall back to walking
        return None
    return matches[0] if matches else None


def _focused_field(pid):
    import gi

    gi.require_version("Atspi", "2.0")
    from gi.repository import Atspi

    desktop = Atspi.get_desktop(0)
    apps = [desktop.get_child_at_index(i) for i in range(desktop.get_child_count())]
    app = next((a for a in apps if a is not None and a.get_process_id() == pid), None)
    if app is None:
        return None, "the app isn't visible to accessibility"

    # The window the user is typing in first, then the app's other windows.
    windows = [app.get_child_at_index(i) for i in range(app.get_child_count())]
    windows = [w for w in windows if w is not None]
    active = [w for w in windows if w.get_state_set().contains(Atspi.StateType.ACTIVE)]
    for window in active + [w for w in windows if w not in active]:
        hit = _search(window, Atspi)
        if hit is not None:
            return hit, ""

    started = time.monotonic()

    def walk(node, depth):
        if node is None or depth > 40 or time.monotonic() - started > WALK_LIMIT_S:
            return None
        states = node.get_state_set()
        if states.contains(Atspi.StateType.FOCUSED) and states.contains(Atspi.StateType.EDITABLE):
            return node
        for i in range(min(node.get_child_count(), 200)):
            hit = walk(node.get_child_at_index(i), depth + 1)
            if hit is not None:
                return hit
        return None

    field = walk(app, 0)
    if field is None:
        return None, "no focused text field found"
    return field, ""


def _atspi():
    import gi

    gi.require_version("Atspi", "2.0")
    from gi.repository import Atspi

    return Atspi


def _active_window(pid):
    """(window index, window) of the app's active window, or (None, why)."""
    Atspi = _atspi()
    desktop = Atspi.get_desktop(0)
    apps = [desktop.get_child_at_index(i) for i in range(desktop.get_child_count())]
    app = next((a for a in apps if a is not None and a.get_process_id() == pid), None)
    if app is None:
        return None, "the app isn't visible to accessibility"
    windows = [(i, app.get_child_at_index(i)) for i in range(app.get_child_count())]
    windows = [(i, w) for i, w in windows if w is not None]
    if not windows:
        return None, "the app has no windows"
    active = [(i, w) for i, w in windows if w.get_state_set().contains(Atspi.StateType.ACTIVE)]
    return (active or windows)[0], ""


def _editable_under(window, Atspi):
    """Editable, showing objects under `window` (the app's search first, else a walk)."""
    try:
        states = Atspi.StateSet.new([Atspi.StateType.EDITABLE, Atspi.StateType.SHOWING])
        rule = Atspi.MatchRule.new(
            states, Atspi.CollectionMatchType.ALL,
            {}, Atspi.CollectionMatchType.ALL,
            [], Atspi.CollectionMatchType.ALL,
            [], Atspi.CollectionMatchType.ALL,
            False,
        )
        matches = list(Atspi.Collection.get_matches(window, rule, Atspi.CollectionSortOrder.CANONICAL, 0, True))
    except Exception:  # noqa: BLE001 - not supported here: walk instead
        matches = None
    if matches is not None:
        return matches
    found, started = [], time.monotonic()

    def walk(node, depth):
        if node is None or depth > 40 or time.monotonic() - started > WALK_LIMIT_S:
            return
        states = node.get_state_set()
        if states.contains(Atspi.StateType.EDITABLE) and states.contains(Atspi.StateType.SHOWING):
            found.append(node)
        for i in range(min(node.get_child_count(), 200)):
            walk(node.get_child_at_index(i), depth + 1)

    walk(window, 0)
    return found


def _path(node, window):
    """Child indexes from `window` down to `node` (None if it isn't under it)."""
    path = []
    while node is not None and node != window:
        index = node.get_index_in_parent()
        if index < 0:
            return None
        path.append(index)
        node = node.get_parent()
    return list(reversed(path)) if node == window else None


def _raw_text(node, Atspi):
    """(text with placeholders, character count); text None when it can't be read."""
    if node.get_text_iface() is None:
        return None, 0
    length = Atspi.Text.get_character_count(node)
    if length <= TEXT_HEAD + TEXT_TAIL:
        text = Atspi.Text.get_text(node, 0, length)
    else:
        text = Atspi.Text.get_text(node, 0, TEXT_HEAD) + "\n…\n" + Atspi.Text.get_text(node, length - TEXT_TAIL, length)
    return text, length


# --- blank boxes ----------------------------------------------------------------------
# A copy of clicky/writing/blank.py's `invisible` (this program can't import Clicky's code;
# a test keeps the two the same).
INVISIBLE_CATEGORIES = frozenset({"Zs", "Zl", "Zp", "Cc", "Cf", "Mn", "Me"})
BLANK_LETTERS = frozenset({"ᅟ", "ᅠ", "ㅤ", "ﾠ", "⠀", "᠎"})
# Objects behind a placeholder that may be an empty block (an empty line or paragraph of a
# rich-text box). Anything else (an image, link, button, list item, …) is content.
EMPTY_BLOCK_ROLES = frozenset({"section", "paragraph", "panel", "static", "text"})
OBJECT_DEPTH, OBJECT_LIMIT = 4, 40
CODES_LIMIT, ROLES_LIMIT = 32, 8


def invisible(ch):
    import unicodedata

    return ch.isspace() or ch in BLANK_LETTERS or unicodedata.category(ch) in INVISIBLE_CATEGORIES


def _blank(text):
    """No visible character (placeholders aside)."""
    return all(invisible(ch) for ch in text if ch != PLACEHOLDER)


def _embedded(node, Atspi):
    """The objects behind `node`'s placeholders, in order (None: they can't be told)."""
    try:
        if node.get_hypertext_iface() is not None:
            links = [Atspi.Hypertext.get_link(node, i) for i in range(Atspi.Hypertext.get_n_links(node))]
            return [link.get_object(0) if link is not None else None for link in links]
        return [node.get_child_at_index(i) for i in range(node.get_child_count())]
    except Exception:  # noqa: BLE001 - can't be told: content
        return None


def _blocks_empty(node, raw, Atspi, roles, budget, depth=0):
    """Every placeholder in `raw` (the text of `node`) stands for an empty block, checked down
    to OBJECT_DEPTH levels and OBJECT_LIMIT objects. Anything unclear is content (False)."""
    objects = _embedded(node, Atspi)
    if objects is None or len(objects) != raw.count(PLACEHOLDER):
        return False
    for obj in objects:
        budget[0] -= 1
        if obj is None or budget[0] < 0 or depth >= OBJECT_DEPTH:
            return False
        try:
            role = obj.get_role_name() or "unknown"
            text, length = _raw_text(obj, Atspi)
        except Exception:  # noqa: BLE001
            return False
        roles.append(role)
        if role not in EMPTY_BLOCK_ROLES or text is None or length > TEXT_HEAD + TEXT_TAIL or not _blank(text):
            return False
        if PLACEHOLDER in text and not _blocks_empty(obj, text, Atspi, roles, budget, depth + 1):
            return False
    return True


def blank_details(node, raw, length, Atspi):
    """For a box with no visible text and a count above 0: its placeholders, whether they are all
    empty blocks, the object roles and the character codes (never text). {} for any other box."""
    if raw is None or length <= 0 or length > TEXT_HEAD + TEXT_TAIL or not _blank(raw):
        return {}
    placeholders = raw.count(PLACEHOLDER)
    roles = []
    ok = placeholders == 0 or _blocks_empty(node, raw, Atspi, roles, [OBJECT_LIMIT])
    return {
        "placeholders": placeholders,
        "blocks_empty": bool(ok),
        "objects": roles[:ROLES_LIMIT],
        "codes": [ord(ch) for ch in raw[:CODES_LIMIT]],
    }


def _describe(node, Atspi, field_id):
    states = node.get_state_set()
    try:
        box = Atspi.Component.get_extents(node, Atspi.CoordType.SCREEN)
        x, y, w, h = box.x, box.y, box.width, box.height
    except Exception:  # noqa: BLE001 - no position: still listed
        x = y = w = h = -1
    raw, length = _raw_text(node, Atspi)
    text = raw.replace(PLACEHOLDER, " ") if raw is not None else None
    described = {
        "id": field_id,
        "role": node.get_role_name(),
        "name": node.get_name() or "",
        "description": node.get_description() or "",
        "multi_line": states.contains(Atspi.StateType.MULTI_LINE),
        "focused": states.contains(Atspi.StateType.FOCUSED),
        "x": x, "y": y, "w": w, "h": h,
        "readable": text is not None,
        "text": text,
        "length": length,
    }
    described.update(blank_details(node, raw, length, Atspi))
    return described


def list_fields(pid):
    Atspi = _atspi()
    found, why = _active_window(pid)
    if found is None:
        return {"ok": False, "why": why}
    window_index, window = found
    nodes = _editable_under(window, Atspi)
    fields = []
    for node in nodes:
        path = _path(node, window)
        if path is None:
            continue
        # The paragraphs of a rich-text box are editable too: keep only the outer box.
        parent, nested = node.get_parent(), False
        while parent is not None and parent != window:
            if parent in nodes:
                nested = True
                break
            parent = parent.get_parent()
        if nested:
            continue
        fields.append(_describe(node, Atspi, f"{window_index}:" + ".".join(str(i) for i in path)))
        if len(fields) >= FIELD_LIMIT:
            break
    fields.sort(key=lambda f: (f["y"], f["x"]))  # reading order: top to bottom, then left to right
    return {"ok": True, "window": window.get_name() or "", "fields": fields}


def _resolve(pid, field_id):
    Atspi = _atspi()
    found, why = _active_window(pid)
    if found is None:
        return None, why
    window_index, window = found
    wanted_window, _, path = field_id.partition(":")
    if wanted_window != str(window_index):
        return None, "that window isn't the active one any more"
    node = window
    for part in [p for p in path.split(".") if p]:
        node = node.get_child_at_index(int(part)) if node is not None else None
    if node is None or not node.get_state_set().contains(Atspi.StateType.EDITABLE):
        return None, "that field isn't there any more"
    return node, ""


def focus_field(pid, field_id):
    Atspi = _atspi()
    node, why = _resolve(pid, field_id)
    if node is None:
        return {"ok": False, "why": why}
    try:
        Atspi.Component.grab_focus(node)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "why": f"focus refused ({exc.__class__.__name__})"}
    end = time.monotonic() + FOCUS_WAIT_S
    while time.monotonic() < end:
        if node.get_state_set().contains(Atspi.StateType.FOCUSED):
            return {"ok": True}
        time.sleep(0.05)
    return {"ok": False, "why": "focus didn't move to that field"}


def _action_names(node, Atspi):
    try:
        if node.get_action_iface() is None:
            return []
        return [(Atspi.Action.get_action_name(node, i) or "").lower() for i in range(Atspi.Action.get_n_actions(node))]
    except Exception:  # noqa: BLE001 - no actions
        return []


def _extra_text(node):
    """Other accessibility text of an object (its attributes), for the forbidden-word check."""
    try:
        attributes = node.get_attributes() or {}
    except Exception:  # noqa: BLE001
        return ""
    return " ".join(f"{k}: {v}" for k, v in attributes.items() if isinstance(v, str))[:300]


def kind_of(node, Atspi):
    """"text" (an editable box), "control" (something that can be pressed) or "other"."""
    states = node.get_state_set()
    if states.contains(Atspi.StateType.EDITABLE):
        return "text"
    if node.get_role_name() in PRESSABLE_ROLES:
        return "control"
    if any(name in PRESS_ACTIONS for name in _action_names(node, Atspi)):
        return "control"
    return "other"


def describe_button(node, Atspi, button_id):
    try:
        box = Atspi.Component.get_extents(node, Atspi.CoordType.SCREEN)
        x, y, w, h = box.x, box.y, box.width, box.height
    except Exception:  # noqa: BLE001
        x = y = w = h = -1
    return {
        "id": button_id, "role": node.get_role_name(), "name": node.get_name() or "",
        "description": node.get_description() or "", "extra": _extra_text(node),
        "x": x, "y": y, "w": w, "h": h,
    }


def _pressables_under(window, Atspi):
    """Objects under the window that could be pressed (showing), the app's search first, else a walk."""
    found, started = [], time.monotonic()

    def walk(node, depth):
        if node is None or depth > 40 or len(found) >= BUTTON_SCAN_LIMIT:
            return
        if time.monotonic() - started > BUTTON_WALK_LIMIT_S:
            return
        states = node.get_state_set()
        if not states.contains(Atspi.StateType.SHOWING) and depth > 0:
            return
        if depth > 0 and kind_of(node, Atspi) == "control":
            found.append(node)
        for i in range(min(node.get_child_count(), 200)):
            walk(node.get_child_at_index(i), depth + 1)

    walk(window, 0)
    return found


def list_buttons(pid):
    Atspi = _atspi()
    found, why = _active_window(pid)
    if found is None:
        return {"ok": False, "why": why}
    window_index, window = found
    try:
        frame = Atspi.Component.get_extents(window, Atspi.CoordType.SCREEN)
        bounds = (frame.x, frame.y, frame.x + frame.width, frame.y + frame.height)
    except Exception:  # noqa: BLE001 - no frame: positions aren't filtered
        bounds = None
    buttons = []
    for node in _pressables_under(window, Atspi):
        states = node.get_state_set()
        if not states.contains(Atspi.StateType.VISIBLE) or not states.contains(Atspi.StateType.SENSITIVE):
            continue
        path = _path(node, window)
        if path is None:
            continue
        item = describe_button(node, Atspi, f"{window_index}:" + ".".join(str(i) for i in path))
        if item["w"] <= 0 or item["h"] <= 0:
            continue
        if bounds is not None and not (bounds[0] <= item["x"] + item["w"] // 2 < bounds[2]
                                       and bounds[1] <= item["y"] + item["h"] // 2 < bounds[3]):
            continue  # scrolled out of the window
        buttons.append(item)
    buttons.sort(key=lambda b: (b["y"], b["x"]))
    return {"ok": True, "buttons": buttons[:BUTTON_LIMIT]}


def _node_at_path(pid, node_id):
    """The object with that id in the app's active window (any kind), or (None, why)."""
    found, why = _active_window(pid)
    if found is None:
        return None, why
    window_index, window = found
    wanted_window, _, path = node_id.partition(":")
    if wanted_window != str(window_index):
        return None, "that window isn't the active one any more"
    node = window
    for part in [p for p in path.split(".") if p]:
        node = node.get_child_at_index(int(part)) if node is not None else None
    if node is None:
        return None, "that button isn't there any more"
    return node, ""


def press(pid, node_id, expected_name, Atspi=None, node=None):
    """Press one button, re-checked first: same name, showing, enabled, not a text box."""
    Atspi = Atspi or _atspi()
    if node is None:
        node, why = _node_at_path(pid, node_id)
        if node is None:
            return {"ok": False, "why": why}
    name = node.get_name() or ""
    if not name or name != expected_name:
        return {"ok": False, "why": "that button changed, so it wasn't pressed"}
    states = node.get_state_set()
    if not states.contains(Atspi.StateType.SHOWING) or not states.contains(Atspi.StateType.SENSITIVE):
        return {"ok": False, "why": "that button can't be pressed right now"}
    if states.contains(Atspi.StateType.EDITABLE):
        return {"ok": False, "why": "that's a text box, not a button"}
    names = _action_names(node, Atspi)
    index = next((i for i, n in enumerate(names) if n in PRESS_ACTIONS), 0 if names else None)
    if index is None:
        return {"ok": False, "why": "that button has no action to press"}
    try:
        done = Atspi.Action.do_action(node, index)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "why": f"press refused ({exc.__class__.__name__})"}
    return {"ok": bool(done), "why": "" if done else "press refused"}


def spot(node, Atspi):
    """What a point shows: the deepest object there, or the first text box or control above it
    (a label inside a button counts as the button)."""
    current, depth = node, 0
    while current is not None and depth < 12:
        kind = kind_of(current, Atspi)
        if kind != "other":
            break
        parent = current.get_parent()
        if parent is None or parent.get_role_name() in ("frame", "window", "application"):
            break
        current, depth = parent, depth + 1
    target = current if current is not None and kind_of(current, Atspi) != "other" else node
    return {
        "ok": True, "found": True, "kind": kind_of(target, Atspi), "role": target.get_role_name(),
        "name": target.get_name() or "", "description": target.get_description() or "", "extra": _extra_text(target),
    }


def what_is_at(pid, x, y):
    Atspi = _atspi()
    desktop = Atspi.get_desktop(0)
    apps = [desktop.get_child_at_index(i) for i in range(desktop.get_child_count())]
    app = next((a for a in apps if a is not None and a.get_process_id() == pid), None)
    if app is None:
        return {"ok": True, "found": False, "why": "the app isn't visible to accessibility"}
    windows = [w for w in (app.get_child_at_index(i) for i in range(app.get_child_count())) if w is not None]
    active = [w for w in windows if w.get_state_set().contains(Atspi.StateType.ACTIVE)]
    for window in active + [w for w in windows if w not in active]:
        try:
            node = Atspi.Component.get_accessible_at_point(window, x, y, Atspi.CoordType.SCREEN)
        except Exception:  # noqa: BLE001
            node = None
        if node is None:
            continue
        for _ in range(40):
            try:
                child = Atspi.Component.get_accessible_at_point(node, x, y, Atspi.CoordType.SCREEN)
            except Exception:  # noqa: BLE001
                child = None
            if child is None or child == node:
                break
            node = child
        return spot(node, Atspi)
    return {"ok": True, "found": False}


def main():
    command, pid = sys.argv[1], int(sys.argv[2])
    if command == "fields":
        print(json.dumps(list_fields(pid)))
        return
    if command == "buttons":
        print(json.dumps(list_buttons(pid)))
        return
    if command == "press":
        print(json.dumps(press(pid, sys.argv[3], sys.argv[4])))
        return
    if command == "at":
        print(json.dumps(what_is_at(pid, int(sys.argv[3]), int(sys.argv[4]))))
        return
    if command == "focus":
        print(json.dumps(focus_field(pid, sys.argv[3])))
        return
    field, why = _focused_field(pid)
    if field is None:
        print(json.dumps({"ok": False, "why": why}))
        return
    from gi.repository import Atspi

    if command == "focused":
        print(json.dumps({"ok": True, "field": _describe(field, Atspi, "")}))
        return

    # Call the Text methods through the interface class: `field.get_text()` is an
    # unrelated, deprecated Accessible method with the same name.
    readable = field.get_text_iface() is not None
    if command == "read":
        if not readable:
            print(json.dumps({"ok": False, "why": "the field can't be read"}))
            return
        text = Atspi.Text.get_text(field, 0, -1)
        if PLACEHOLDER in text:
            print(json.dumps({"ok": False, "why": "the field's text can't be read as plain text"}))
            return
        editable = field.get_editable_text_iface() is not None
        print(json.dumps({"ok": True, "text": text, "editable": editable}))
        return
    if command == "select":
        if not readable:
            print(json.dumps({"ok": False, "why": "the field can't be read"}))
            return
        start, end = int(sys.argv[3]), int(sys.argv[4])
        if Atspi.Text.get_n_selections(field) > 0:
            done = Atspi.Text.set_selection(field, 0, start, end)
        else:
            done = Atspi.Text.add_selection(field, start, end)
        time.sleep(0.1)
        if not done or Atspi.Text.get_n_selections(field) < 1:
            print(json.dumps({"ok": False, "why": "selection refused"}))
            return
        got = Atspi.Text.get_selection(field, 0)
        selected = Atspi.Text.get_text(field, got.start_offset, got.end_offset)
        print(json.dumps({"ok": True, "text": selected}))
        return
    editable = field.get_editable_text_iface()
    if editable is None:
        print(json.dumps({"ok": False, "why": "the field doesn't accept direct insert"}))
        return
    text = sys.stdin.read()
    if command == "insert":
        where = sys.argv[3] if len(sys.argv) > 3 else "caret"
        if where == "start":
            offset = 0
        elif where == "end":
            if not readable:
                print(json.dumps({"ok": False, "why": "the field's end can't be found"}))
                return
            offset = Atspi.Text.get_character_count(field)
        else:
            offset = Atspi.Text.get_caret_offset(field) if readable else 0
        ok = editable.insert_text(offset, text, len(text.encode("utf-8")))
        print(json.dumps({"ok": bool(ok), "offset": offset, "length": len(text), "why": "" if ok else "insert refused"}))
        return
    if command == "replace":
        offset, length = int(sys.argv[3]), int(sys.argv[4])
        deleted = editable.delete_text(offset, offset + length)
        inserted = deleted and editable.insert_text(offset, text, len(text.encode("utf-8")))
        print(json.dumps({"ok": bool(inserted), "why": "" if inserted else "replace refused"}))
        return
    print(json.dumps({"ok": False, "why": f"unknown command {command}"}))


if __name__ == "__main__":
    main()
