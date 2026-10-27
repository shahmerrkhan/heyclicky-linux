# Clicky for Linux

> A Linux port of [**clicky.so**](https://clicky.so) — the original macOS Clicky by [@farzaa](https://github.com/farzaa/clicky).
> This project is **dedicated to the original Clicky**. All credit for the idea, the design, and the vibe goes to the original author; this repository is just a gift to the Linux community so we can play with the same thing.
> Built from scratch for X11 + PyQt6, with the macOS app's behavior as the reference.

Clicky is the little AI buddy that lives next to your cursor. Hit a hotkey, ask a question about whatever is on your screen, and a streaming bubble answers right there — and if the model points to something, a blue cursor flies over and taps it.

Clicky has two modes you can flip between in the input bar:

- **Vision mode** (default): screenshot + question → Gemini (called directly; add `/p` anywhere in a question for the strong model or `/s` to add Tavily web search results) → answer streams into a bubble, optionally pointing at something on screen.
- **Code mode**: prompt → spawns `claude -p` ([Claude Code](https://docs.anthropic.com/en/docs/claude-code)) as a subprocess in a configured workspace → its progress (text + tool calls) streams into the same bubble. Lets you ask Clicky to *do* things in a directory, not just describe what it sees.

## Quick start

```bash
# 1. Check system dependencies
make check-system

# 2. (If missing) install the recommended apt packages
sudo apt install -y python3-venv libxcb-cursor0 libxkbcommon0 libegl1 libgl1 libfontconfig1 libdbus-1-3

# 3. Install Python deps (uses uv)
make install

# 4. Add your API keys
cp .env.example .env
# Open .env in your editor and set GEMINI_API_KEY=... (https://aistudio.google.com/apikey)
# and, for /s web search, TAVILY_API_KEY=... (https://app.tavily.com, free plan)

# 5. Run it
make run
```

No proxy or server is needed: the app calls the Gemini API directly. (The `worker/` folder is from the old OpenAI setup and isn't used by vision mode anymore.)

A blue triangle icon shows up in the system tray. Without a `GEMINI_API_KEY` or `TAVILY_API_KEY` Clicky still starts (Code mode works); a tray notification tells you what's off until the key is set. Press **Ctrl+Alt+Space** to open the input.

## Fast model, strong model, and `/p`

Every question goes to the **fast model** first (`GEMINI_FAST_MODEL`, default `gemini-3.5-flash-lite`, medium thinking) with a 10-second first-word timer. If it fails, the **strong model** (`GEMINI_MODEL`, default `gemini-3.8-flash`, medium thinking) takes over with a 20-second timer. They alternate fast → strong for 3 rounds, with a 3-second pause between rounds, and then Clicky shows an error.

- Each try gets one retry after 2 seconds if the model says it's busy.
- Busy, a used-up limit, too slow (no first word in time), and an empty answer all move on to the other model. A blocked answer (safety filter) is shown with its reason. A bad key, unknown model, or bad request stops right away.
- A model that reports its **daily** limit used up is skipped until midnight Pacific, so later questions go straight to the other one. If both are at their daily limits you get the error immediately.
- Add **`/p`** anywhere in a question to ask the strong model first, with a 90-second timer. If it fails, the footer says why and the fast model answers instead (no alternating after that).
- Commands: `/p` (strong) and `/s` (search) work anywhere in the question, in any order and any case — `who won the cup /p /s`, `what is /s the latest python`, `/ps q`, `/Sp q`. Each is removed before the question goes to Gemini or Tavily. A command must stand alone: with a space or the start/end of the text on each side, so `/p,`, `(/s)`, `/usr/bin/python`, `x.com/p` and `10/27` don't count. Anything inside a closed pair of double quotes or backticks is left alone (`"xcopy /s"`), an unclosed quote protects nothing, and `//s` types a literal `/s`. A small **strong** / **search** chip in the input bar shows what the typed commands will do. Commands only apply in vision mode.
- The footer always names the model that answered, e.g. `answered by the strong model (gemini-3.8-flash) after 3 tries — the fast model was busy`. While switching, the bubble shows `trying <model>…`.
- Once an answer starts it is never cut off. Closing the bubble cancels everything in flight. Both models share one conversation history.

## Web search (`/s`)

Add `/s` anywhere in a question (for example `latest python release /s`) to search the web first. The question, with the commands removed, goes to [Tavily](https://tavily.com) as typed (advanced depth, 10 results, 2 passages per site); the results go to Gemini as untrusted reference text, and a **sources:** line with one link per site appears under the answer. Links open in the browser set by `LINK_BROWSER_COMMAND` (default `opera-gx`).

Without `TAVILY_API_KEY`, or when the month's credits are used up, `/s` questions are answered without search and the bubble footer says why.

## Usage estimates

Under every vision answer:

- `today (estimate): 12 questions, 3 with search · resets at midnight pacific` — successful answers only (failed, blocked, and cut-off ones don't count; a retry or model switch never counts twice). Google only shows the real numbers in [AI Studio](https://aistudio.google.com).
- `search credits this month (estimate): 46 / 1000 · resets on nov 1` — Tavily credits used by successful searches (2 per search), reset on the 1st.

Both are saved to `~/.local/state/clicky/usage.json`.

## The response bubble

- Long answers scroll inside the bubble (auto-follows the streaming bottom, stays put when you scroll up to re-read).
- Text is selectable — drag across it to copy.
- **Drag** the bubble by clicking anywhere on its padding (any spot that isn't the text, scrollbar, or × button). The cursor turns into a move handle when you can grab.
- **Resize** from the bottom-right corner (the cursor becomes a diagonal arrow there). Once you've resized, the bubble keeps its size for that response. Sending a new prompt resets it back to auto-fit.

## Code mode (Claude Code agent)

Click the small `vision` / `code` pill on the left of the input bar to switch. In Code mode:

- Your prompt is handed to a `claude -p` subprocess in the workspace shown under the input. Click the workspace chip to pick a different directory.
- For one-off overrides, start your prompt with `@/path/to/dir ` (or `@"/path with spaces" `) — that directory is used as the cwd for that single call.
- The bubble streams Claude Code's prose and shows compact tool chips (`› Read foo.py`, `✎ Edit bar.py`, etc.) so you can see what it's doing.
- **Shell commands are blocked by default**. Flip the `shell: off` pill to `shell: on` to allow `Bash` — you'll get a confirmation dialog the first time, because this opens the door to `rm`, `git push`, package installs, and anything else the model decides to run. Toggle off when you don't need it.
- File edits go through with `--permission-mode acceptEdits`; check `git diff` if you want to review what changed.

Requirements: [Claude Code](https://docs.anthropic.com/en/docs/claude-code) installed and authenticated (`claude auth`). Clicky calls the `claude` binary from PATH; override with `CLAUDE_CODE_PATH` in `.env`.

## Architecture (short version)

```
Hotkey (pynput, X11)
   ↓
Overlay (transparent fullscreen PyQt6 window)
   ↓ user types and presses Enter
Screenshot (mss, all monitors)
   ↓
(/s only) Tavily web search → results as untrusted text
   ↓
Gemini API (google-genai SDK): fast model first (10 s first-word timer),
   ↓ fails → strong model (20 s), alternating 3 rounds · /p → strong (90 s) then fast
   ↓ streamed chunks
Bubble UI (text streaming, then sources + notes + usage footer)
   ↓ [POINT:x,y] if present (0-1000 scale, x across then y down)
Convert to screen pixels (clamped) → blue cursor bezier animation
```

| Part | File | macOS counterpart |
|---|---|---|
| State machine | `clicky/state.py` | `CompanionManager.swift` |
| LLM streaming client | `clicky/llm/gemini.py` | `ClaudeAPI.swift` |
| Fast/strong model order, retry, fallback | `clicky/llm/model_router.py` | (new) |
| `/p` and `/s` prefixes | `clicky/llm/prefixes.py` | (new) |
| Web search (Tavily) | `clicky/llm/web_search.py` | (new) |
| POINT parser | `clicky/llm/point_parser.py` | `CompanionManager.parsePointingCoordinates` |
| Screenshot | `clicky/screen/capture.py` | `CompanionScreenCaptureUtility.swift` |
| Hotkey | `clicky/input/hotkey.py` | `GlobalPushToTalkShortcutMonitor.swift` |
| Tray | `clicky/ui/tray.py` | `MenuBarPanelManager.swift` |
| Panel | `clicky/ui/panel.py` | `CompanionPanelView.swift` |
| Overlay + cursor | `clicky/ui/overlay.py`, `cursor.py` | `OverlayWindow.swift` |
| Response bubble | `clicky/ui/response_bubble.py` | `CompanionResponseOverlay.swift` |
| Model → screen coordinate conversion | `CapturedScreen.global_point_from_normalized` in `clicky/screen/capture.py` | — |
| Usage estimates (daily Gemini, monthly Tavily) | `clicky/usage_store.py` | (new) |
| Worker proxy (legacy, unused by vision mode) | `worker/src/index.ts` | same idea, OpenAI-flavored |
| Claude Code agent | `clicky/llm/claude_code.py` | (new — no macOS counterpart yet) |
| Mode + workspace persistence | `clicky/state_store.py` | (new — no macOS counterpart yet) |

## Linux-specific choices

- **X11 + GNOME** is the primary target. On Wayland, `pynput` global hotkeys don't work reliably on most setups — that path will come later.
- **Default hotkey is `Ctrl+Alt+Space`** (modifier-only `Ctrl+Alt` doesn't trigger reliably on Linux). Override it in `.env`.
- **GNOME 45+**: you need the [AppIndicator and KStatusNotifierItem Support](https://extensions.gnome.org/extension/615/appindicator-support/) extension for the tray icon to show up.
- **Screenshot downscaling**: each monitor is resized to a 1600 px long edge before sending to keep payloads small. The model points on a 0-1000 scale, so the downscale never affects where the cursor lands.
- **Cursor**: we don't touch the real system cursor (forbidden on Wayland, rude on X11). Instead we draw our own blue triangle on a transparent overlay — the original Clicky does the same.

## Development

```bash
make dev             # Run the app with DEBUG logging
```

## Roadmap

- [x] Code mode — Claude Code subprocess as a Clicky backend
- [ ] Native GUI tools (click, type, scroll) via Gemini tool-calling, with per-action confirmation bubbles
- [ ] Per-tool confirmation in Code mode via Claude Code `PreToolUse` hooks (IPC socket)
- [ ] Microphone + AssemblyAI streaming (push-to-talk)
- [ ] ElevenLabs TTS playback
- [ ] Wayland fallback (XDG Portal screenshot, evdev hotkey)
- [ ] Onboarding flow
- [ ] PyInstaller single-file binary
- [ ] Conversation history persistence

## Credits

- Original idea, design, and macOS implementation: [**clicky.so**](https://clicky.so) by [@farzaa](https://github.com/farzaa/clicky). Go look at the original — it's the real deal.
- This repo: a from-scratch Linux re-implementation in Python/PyQt6, written as a tribute so Linux users can run something close to the same experience.

## License

MIT (matches the original Clicky).
