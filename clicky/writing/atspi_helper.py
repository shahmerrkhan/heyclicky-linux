"""AT-SPI helper for writing mode. Runs as a separate program with the system
Python (it needs the GObject bindings, which Clicky's own environment doesn't
have): `/usr/bin/python3 -I atspi_helper.py <command> <pid>`, text on stdin.

Commands (each prints one JSON line):
  insert <pid>   insert stdin text at the caret of the focused editable field
                 → {"ok": true, "offset": <caret>, "length": <chars>}
  read <pid>     the focused field's whole text (for safe draft replacement),
                 and whether it can be edited directly (native apps) or only
                 read (Chrome, Opera) → {"ok": true, "text": "...", "editable": bool}.
                 Text that comes back with object placeholders (U+FFFC) can't
                 be read as plain text, so it counts as unreadable.
  replace <pid> <offset> <length>
                 delete <length> chars at <offset>, then insert stdin text there
                 → {"ok": true}

Failures print {"ok": false, "why": "..."} and exit 0, so Clicky can move on
to the next method.
"""

import json
import sys
import time

WALK_LIMIT_S = 5.0  # the fallback walk only; the built-in search is much faster
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


def main():
    command, pid = sys.argv[1], int(sys.argv[2])
    field, why = _focused_field(pid)
    if field is None:
        print(json.dumps({"ok": False, "why": why}))
        return
    from gi.repository import Atspi

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
    editable = field.get_editable_text_iface()
    if editable is None:
        print(json.dumps({"ok": False, "why": "the field doesn't accept direct insert"}))
        return
    text = sys.stdin.read()
    if command == "insert":
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
