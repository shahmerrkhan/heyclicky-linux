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
                 A field inside another editable field (the paragraphs of a
                 rich-text box) is left out: the outer box is the field.
  focus <pid> <id>
                 move focus to that field (an id from `fields`) and confirm it
                 → {"ok": true}
  focused <pid>  the editable field that has focus (after a click), described
                 like one entry of `fields` (its "id" is empty)
                 → {"ok": true, "field": {...}}

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


def _field_text(node, Atspi):
    if node.get_text_iface() is None:
        return None, 0
    length = Atspi.Text.get_character_count(node)
    if length <= TEXT_HEAD + TEXT_TAIL:
        text = Atspi.Text.get_text(node, 0, length)
    else:
        text = Atspi.Text.get_text(node, 0, TEXT_HEAD) + "\n…\n" + Atspi.Text.get_text(node, length - TEXT_TAIL, length)
    return text.replace(PLACEHOLDER, " "), length


def _describe(node, Atspi, field_id):
    states = node.get_state_set()
    try:
        box = Atspi.Component.get_extents(node, Atspi.CoordType.SCREEN)
        x, y, w, h = box.x, box.y, box.width, box.height
    except Exception:  # noqa: BLE001 - no position: still listed
        x = y = w = h = -1
    text, length = _field_text(node, Atspi)
    return {
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


def main():
    command, pid = sys.argv[1], int(sys.argv[2])
    if command == "fields":
        print(json.dumps(list_fields(pid)))
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
