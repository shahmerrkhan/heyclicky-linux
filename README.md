# Clicky for Linux

> A Linux port of [**clicky.so**](https://clicky.so) — the original macOS Clicky by [@farzaa](https://github.com/farzaa/clicky).
> This project is **dedicated to the original Clicky**. All credit for the idea, the design, and the vibe goes to the original author; this repository is just a gift to the Linux community so we can play with the same thing.
> Built from scratch for X11 + PyQt6, with the macOS app's behavior as the reference.

Clicky is the little AI buddy that lives next to your cursor. Hit a hotkey, ask a question about whatever is on your screen, and a streaming bubble answers right there — and if the model points to something, a blue cursor flies over and taps it.

Clicky has three modes you can flip between in the input bar (it always starts in Vision):

- **Vision mode** (default): screenshot + question → Gemini (called directly; the **strong** button asks the strong model first, **search** adds Tavily web search results) → answer streams into a bubble, optionally pointing at something on screen.
- **Write mode**: messy thoughts → a ready-to-send message, shown in the bubble and typed into the box you were in (see [Writing mode](#writing-mode)).
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
# and, for web search, TAVILY_API_KEY=... (https://app.tavily.com, free plan)

# 5. Run it
make run
```

No proxy or server is needed: the app calls the Gemini API directly. (The `worker/` folder is from the old OpenAI setup and isn't used by vision mode anymore.)

Clicky's icon shows up in the system tray. Without a `GEMINI_API_KEY` or `TAVILY_API_KEY` Clicky still starts (Code mode works); a tray notification tells you what's off until the key is set. Press **Ctrl+Alt+Space** to open the input.

## Run it as a desktop app

```bash
make desktop-install     # app-menu entry "Clicky" + start at login (5 s after the desktop is up)
make desktop-uninstall   # removes both again
```

This writes two small files, with the repo's path filled in at install time: `~/.local/share/applications/clicky.desktop` (the app menu) and `~/.config/autostart/clicky.desktop` (login). Both run `scripts/clicky-launch`, which starts `.venv/bin/python -m clicky` from the repo with no terminal. It still reads `.env` and uses the project's `.venv`. Neither file is overwritten or removed if it wasn't written by `make desktop-install`.

- **Only one copy runs.** A second start (menu, login, or `make run`) sees the lock in `$XDG_RUNTIME_DIR/clicky.lock`, logs "already running" and exits before touching the talk key or the hotkey.
- **Tray menu:** **Restart** shuts down cleanly and starts a fresh copy (after 5 s it restarts even if the shutdown hangs). **Quit** stops it; it does not come back until you open it again or log in.
- **Crashes:** the launcher restarts Clicky after 2 s, at most 3 times within a minute; after that it gives up and shows one notification. Before a restart it clears keycode 8 if a crash left it bound (writing mode briefly borrows keycode 8 to type some symbols, so a crash mid-symbol can leave it set).
- **Logs** (no terminal): `~/.local/state/clicky/clicky.log` (Clicky, 1 MB x 3 files) and `launcher.log` next to it (starts, exits, restarts, crash traces). From a terminal, `make run` logs to the terminal as before.
- **Stop it:** tray > Quit, or `pkill -TERM -f "python -m clicky"` (a clean stop, no restart). **Don't start it at login:** untick Clicky in Settings > Session and Startup > Application Autostart, or run `make desktop-uninstall`.

## Input bar buttons

There are no typed commands: what you type is sent exactly as typed (`/p` or `//s` is just text). Instead, a second row under the text box has buttons. None of them takes focus, so you keep typing.

```
[Vision|Code|Write]  ask clicky anything…                              ◫ area
[strong] [search] [write]  [extra short|short|medium|long]       [speak: off]
```

| Button | Key | Shown in | What it does |
|---|---|---|---|
| **strong** | Alt+P | Vision | Strong model first (see below) |
| **search** | Alt+S | Vision, Write | Tavily web search first (see [Web search](#web-search)) |
| **write** | Alt+W | Vision | A one-off draft (see [Writing mode](#writing-mode)) |
| **extra short / short / medium / long** | Alt+4 / 5 / 6 / 7 | all modes | Answer length (below) |
| **speak** | Alt+K | all modes | Read typed answers aloud (below) |

- **strong**, **search**, **write** and the length last **one question**: they reset after you send and whenever the bar opens fresh.
- **Length**: one lit at a time; clicking the lit one turns it off. None lit = the normal length. Typed text is never searched for length words; only the buttons set it.

  | | Questions and search | Write drafts | Code: Claude Code's closing report |
  |---|---|---|---|
  | none | normal | normal | normal |
  | extra short | 1 sentence, ≤ ~25 words | 1–2 sentences, ≤ ~35 words | 1 sentence, ≤ ~25 words |
  | short | 3–5 sentences, ~60–110 words | 3–6 sentences, ~60–120 words | 2–4 sentences, ~40–80 words |
  | medium | 2 short paragraphs, ~120–180 words | 2 short paragraphs, ~120–170 words | 1–2 short paragraphs, ~100–160 words |
  | long | 3–6 short paragraphs, ~200–400 words | 2–4 paragraphs, ~200–300 words | 3–5 short paragraphs, ~200–350 words |

  It's one instruction line at the end of the system prompt (Code: `--append-system-prompt`, about the closing report only, never the work itself). No token cap and nothing is cut, so sizes are approximate. The line is never saved into chat memory. In Write mode a length button wins over "make it shorter" in your text.
- **speak** (on/off, remembered across restarts, off on first run) reads a **typed** answer aloud with Piper once it's complete. Answers over 600 spoken characters get a short summary from the fast model instead. Code mode says which files were edited, then Claude's closing report, never code; Write mode says whether the draft was typed, then reads it. If something fails you hear one short failure line. Esc (held while Clicky speaks), × or Fn+F5 stop it; sending a new question cuts it off, opening the input bar doesn't. With voice off or Piper missing, the bubble says why nothing was spoken. Fn+F5 voice answers don't use the switch (see [Voice](#voice-hold-to-talk)), except a Code-mode transcript sent from the bar, which follows it.

## Fast model, strong model, and the strong button

Every question goes to the **fast model** first (`GEMINI_FAST_MODEL`, default `gemini-3.5-flash-lite`, medium thinking) with a 10-second first-word timer. If it fails, the **strong model** (`GEMINI_MODEL`, default `gemini-3.8-flash`, medium thinking) takes over with a 20-second timer. They alternate fast → strong for 3 rounds, with a 3-second pause between rounds, and then Clicky shows an error.

- Each try gets one retry after 2 seconds if the model says it's busy.
- Busy, a used-up limit, too slow (no first word in time), and an empty answer all move on to the other model. A blocked answer (safety filter) is shown with its reason. A bad key, unknown model, or bad request stops right away.
- A model that reports its **daily** limit used up is skipped until midnight Pacific, so later questions go straight to the other one. If both are at their daily limits you get the error immediately.
- Click **strong** (or Alt+P, or say "power mode") to ask the strong model first, with a 90-second timer. If it fails, the footer says why and the fast model answers instead (no alternating after that).
- The footer always names the model that answered, e.g. `answered by the strong model (gemini-3.8-flash) after 3 tries — the fast model was busy`. While switching, the bubble shows `trying <model>…`.
- Once an answer starts it is never cut off. Closing the bubble cancels everything in flight. Both models share one conversation history.

## Web search

Click **search** (or Alt+S, or say "web search") to search the web first. A typed question goes to [Tavily](https://tavily.com) as typed (advanced depth, 10 results, 2 passages per site); the results go to Gemini as untrusted reference text, and a **sources:** line with one link per site appears under the answer. Links open in the browser set by `LINK_BROWSER_COMMAND` (default `opera-gx`).

Without `TAVILY_API_KEY`, or when the month's credits are used up, questions with search on are answered without search and the bubble footer says why.

## Writing mode

Dump messy thoughts — typed or spoken — and get a ready-to-send message, shown in the bubble **and typed straight into the box you were in**, with no confirm step.

- **Starting it**: the **Write** mode (click it or **Alt+3**), the **write** button in Vision (Alt+W, a one-off draft), or **"write mode"** at the start of a voice request (a one-off draft: you stay in Vision or Code). In Write mode, voice goes straight to drafting. **search** still searches first; there's no strong button in Write mode (writing already starts with the strong model).
- **Tone**: professional by default, short (about 2–5 sentences, or what a length button says), plain text, no sign-off unless you ask. Say a different tone in the request ("make it casual") to override it.
- **Model**: the strong model first with `WRITING_STRONG_TIMEOUT_S` (default 30 s) to start, then the fast model.
- **The screen is read only when needed**: typed requests go to the model as words only; if it can't write without seeing the screen ("reply to this"), it asks, and only then is a screenshot taken — of the window you were in only, with Clicky's own windows hidden. Voice requests that say "reply to this", "answer this email" and similar grab that window right away.
- **Where it types**: the window that was active when you pressed Ctrl+Alt+Space or Fn+F5. If you've switched windows since, Clicky switches back first; if that window was closed, nothing is typed. Never into Clicky's own windows.
- **How it types**, first that works: direct insert through accessibility (AT-SPI) → paste (your clipboard text is put back about a second later; if the clipboard holds an image or file, paste is skipped) → key by key (line breaks as Shift+Enter, so chat apps don't send early). If focus moves away while typing key by key, typing stops (at most one 24-character chunk may still land in the new window).
- **Terminals**: line breaks become spaces and Enter is **never** pressed; the bubble explains that pressing Enter would run the text as a command.
- **Esc while typing** stops it; the app you're typing into never sees that Esc (Clicky holds Esc only while typing; the X server releases it even if Clicky crashes). If Esc can't be held, nothing is typed. The bubble stays open: "Stopped after N characters", with the full draft above.
- **Follow-ups** ("make it shorter"): the previous draft is replaced only when Clicky can read the box and the draft is still there exactly as typed — in Chrome and Opera (which need the launcher flag below to be readable) only when the box holds nothing but the draft. Otherwise nothing is touched and the new version is copied to your clipboard. Clicky never deletes anything of yours.
- Whenever a draft isn't typed (or only partly), the full draft is copied to your clipboard and the bubble says why.

Chrome and Opera don't let accessibility insert text, so drafts are pasted there. To let Clicky *read* their text boxes (for follow-ups), start them with `--force-renderer-accessibility=form-controls`.

Clicky always starts in **Vision** mode; the mode you pick isn't saved.

## Voice (hold to talk)

Hold **Fn+F5** (Linux key code 63, i.e. F5), say your question, let go. Clicky transcribes it on your machine, shows **you said: …**, and answers. Answers to spoken questions are read aloud sentence by sentence, in full; Esc stops them. Voice drafts in Write mode stay silent. Typed questions are spoken only with the **speak** switch on.

- **Keywords** at the very start: **"power mode"** = strong model (like the strong button), **"web search"** = search (like the search button), **"write mode"** = a one-off draft (like the write button), in any order. For a voice web search the fast model first turns your transcript plus the screenshot into a short search query (the footer shows `searched for: …`); a typed search uses your text as typed.
- **Length words** anywhere in what you say ("briefly", "in detail", "tl;dr", "walk me through"… full list in `clicky/llm/length.py`) only wake a check; they never set a length on their own. The fast model judges whether each one is an instruction about the answer or part of what you're saying ("tell Sam I'll explain briefly" doesn't count), text only, 3 s. If that fails (busy, limits, timeout, unreadable), the strong model tries (5 s); a model known to be daily-limited is skipped. If both fail, simple rules decide (after "tell X", "say that", "write that" or inside quotes it doesn't count; at the start or end, after "explain", "tell me", "give me", "can you"… or before "please" it does). If the rules can't tell, Vision uses the word anyway and Write and Code use the normal length. The bubble shows "checking length words…" meanwhile (usually under 2 s, worst case about 8 s) and the footer says what was applied: `length: short`, `length: short (checked by simple rules)`, `length: short (not double-checked)`, `length words not checked, so normal length`, or `heard two lengths, so normal length`. The check costs one extra fast-model request, only when a length word is heard, and isn't counted in the questions estimate.
- **Code mode**: the transcript (keywords removed) lands in the input bar so you can fix misheard words; length words light the matching length button (the hint line says how they were read) and you can change it. Enter runs it; the answer is spoken only with the **speak** switch on (edited files, then the closing report), silent otherwise.
- **Stopping**: pressing the talk key again, Esc (from any app), closing the bubble, the input-bar hotkey, the tray's **Stop listening / speaking**, or quitting all stop the recorder, Whisper, Piper and playback.
- **Silence**: nothing heard → "didn't catch that", no Gemini call; silent stretches skip Whisper.
- **No length limit.** Safety nets instead: the key is re-checked once a second (a lost release stops listening), 2 minutes of pure silence stops it, less than 1 GB free disk stops it, a piece Whisper can't finish in 60 s becomes `[part missing]`, and phrases Whisper repeats more than 3 times in a row are collapsed. Each piece's audio is deleted right after it's transcribed; no recording is kept.
- **Memory**: voice and typed questions share one chat memory — the last 5 exchanges in full and a ~150-word summary of older ones, written by the fast model in the background after an answer.
- The tray has a **Voice** on/off switch (remembered); `VOICE_ENABLED=false` in `.env` turns voice off completely.

How it works: `parec` records the default microphone (16 kHz mono). While you talk, the audio is cut into 20–30 s pieces at the quietest moment (so no word is split) and each piece goes through a fresh `whisper-cli` run (base.en, timestamps on, lower CPU priority) — Whisper only holds its ~285 MB while a piece is being transcribed. The screenshot is taken when you let go. Piper (lessac medium voice) starts once per answer and streams raw audio into `pacat`. Whisper and Piper share one slot so they never run at the same time; pressing the talk key stops Piper first.

The talk key is read straight from the keyboard (`/dev/input`, your user must be in the `input` group — no sudo) and F5 is grabbed in X11 so apps don't receive it (Ctrl+F5 etc. still work; browsers' F5 refresh doesn't — use Ctrl+R).

### Installing the voice programs (once, no sudo)

Everything lives in `~/.local/share/clicky/voice/` (`VOICE_DIR`), outside the repo. Whisper (MIT) and Piper (GPL-3.0) run as separate programs.

```bash
V=~/.local/share/clicky/voice
# whisper.cpp, built as one self-contained program, plus the English base model
git clone https://github.com/ggml-org/whisper.cpp /tmp/whisper.cpp
cmake -S /tmp/whisper.cpp -B /tmp/whisper.cpp/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_SERVER=OFF
cmake --build /tmp/whisper.cpp/build -j4 --target whisper-cli
install -Dm755 /tmp/whisper.cpp/build/bin/whisper-cli $V/whisper/bin/whisper-cli
sh /tmp/whisper.cpp/models/download-ggml-model.sh base.en $V/whisper/models
# Piper in its own Python environment, the lessac voice, and the error sound
python3 -m venv $V/piper/venv
$V/piper/venv/bin/pip install piper-tts==1.8.0
$V/piper/venv/bin/python -m piper.download_voices --data-dir $V/piper/voices en_US-lessac-medium
mkdir -p $V/sounds && echo "Error." | $V/piper/venv/bin/python -m piper -m $V/piper/voices/en_US-lessac-medium.onnx -f $V/sounds/error.wav
```

`parec` and `pacat` come with PipeWire's PulseAudio support (`pulseaudio-utils`). If anything is missing, Clicky says so in a tray notice at startup.

## Usage estimates

One quiet line under every vision answer:

`today (estimate): 12 questions, 3 with search, resets 09:00 · search credits: 46 / 1000, back on nov 1`

- Questions count successful answers only (failed, blocked, and cut-off ones don't count; a retry or model switch never counts twice) and reset at midnight Pacific (shown in your local time). Google only shows the real numbers in [AI Studio](https://aistudio.google.com).
- Search credits are the Tavily credits used by successful searches (2 per search), reset on the 1st.

Both are saved to `~/.local/state/clicky/usage.json`.

## The look

Dark glass cards with Noto Sans type, one colour per mode — **Vision** blue, **Code** green, **Write** amber — used on the mode switch and the bubble's tag.

- **Input bar**: a Vision / Code / Write switch (click, or **Alt+1** / **Alt+2** / **Alt+3**), a box that grows with your text up to 8 lines and then scrolls (**Enter** sends, **Shift+Enter** adds a line, **Esc** closes), one chip per typed command (**strong**, **search**, **write**), and the ◫ area button.
- **Listening pill**: while you hold the talk key, a small pill near the cursor shows a red dot and the time.
- **Bubble**: up to 560 px wide, with the mode tag and × at the top, a quiet *you said: …* line for voice questions, the answer, and the footer. It never takes focus, so the window you were in stays focused.

## The response bubble

- Long answers scroll inside the bubble (auto-follows the streaming bottom, stays put when you scroll up to re-read).
- Text is selectable — drag across it to copy.
- **Drag** the bubble by its header or padding (any spot that isn't the answer text, scrollbar, footer links, or × button). The cursor turns into a move handle when you can grab.
- **Resize** from the bottom-right corner (the cursor becomes a diagonal arrow there). Once you've resized, the bubble keeps its size for that response. Sending a new prompt resets it back to auto-fit.

## Code mode (Claude Code agent)

Click **Code** on the input bar's mode switch (or press **Alt+2**) to switch. In Code mode:

- Your prompt is handed to a `claude -p` subprocess in the workspace shown under the input. Click the workspace chip to pick a different directory.
- For one-off overrides, start your prompt with `@/path/to/dir ` (or `@"/path with spaces" `) — that directory is used as the cwd for that single call.
- The bubble streams Claude Code's prose and shows compact tool chips (`› Read foo.py`, `✎ Edit bar.py`, etc.) so you can see what it's doing.
- **Shell commands are blocked by default**. Flip the `shell: off` pill to `shell: on` to allow `Bash` — you'll get a confirmation dialog the first time, because this opens the door to `rm`, `git push`, package installs, and anything else the model decides to run. Toggle off when you don't need it.
- File edits go through with `--permission-mode acceptEdits`; check `git diff` if you want to review what changed.

Requirements: [Claude Code](https://docs.anthropic.com/en/docs/claude-code) installed and authenticated (`claude auth`). Clicky calls the `claude` binary from PATH; override with `CLAUDE_CODE_PATH` in `.env`.

## Architecture (short version)

```
Hotkey (pynput, X11)                 Talk key (evdev, F5 grabbed in X11)
   ↓                                    ↓ hold: parec → 20-30 s pieces → whisper-cli
Overlay (transparent fullscreen PyQt6 window)   ↓ let go: transcript ("power mode" / "web search")
   ↓ user types and presses Enter       ↓ (web search: fast model writes the query)
Screenshot (mss, all monitors)
   ↓
(search button / "web search" only) Tavily web search → results as untrusted text
   ↓
Gemini API (google-genai SDK): fast model first (10 s first-word timer),
   ↓ fails → strong model (20 s), alternating 3 rounds · strong button → strong (90 s) then fast
   ↓ streamed chunks
Bubble UI (text streaming, then sources + notes + usage footer)
   ↓ [POINT:x,y] if present (0-1000 scale, x across then y down)
Convert to screen pixels (clamped) → blue cursor bezier animation
   (voice questions only) Piper reads the answer aloud, sentence by sentence → pacat
```

| Part | File | macOS counterpart |
|---|---|---|
| State machine | `clicky/state.py` | `CompanionManager.swift` |
| LLM streaming client | `clicky/llm/gemini.py` | `ClaudeAPI.swift` |
| Fast/strong model order, retry, fallback | `clicky/llm/model_router.py` | (new) |
| Input bar choices, spoken keywords | `clicky/llm/prefixes.py` | (new) |
| Answer length, voice length check | `clicky/llm/length.py` | (new) |
| Chat memory (last 5 + summary) | `clicky/llm/memory.py` | (new) |
| Voice search query | `clicky/llm/search_query.py` | (new) |
| Talk key | `clicky/input/push_to_talk.py` | (new) |
| Recording, cutting at quiet moments | `clicky/audio/recorder.py` | (new) |
| Speech → text (whisper.cpp) | `clicky/transcription/whisper.py`, `listener.py` | (new) |
| Text → speech (Piper) | `clicky/tts/piper.py` | (new) |
| Web search (Tavily) | `clicky/llm/web_search.py` | (new) |
| POINT parser | `clicky/llm/point_parser.py` | `CompanionManager.parsePointingCoordinates` |
| Screenshot | `clicky/screen/capture.py` | `CompanionScreenCaptureUtility.swift` |
| Hotkey | `clicky/input/hotkey.py` | `GlobalPushToTalkShortcutMonitor.swift` |
| Tray | `clicky/ui/tray.py` | `MenuBarPanelManager.swift` |
| Panel | `clicky/ui/panel.py` | `CompanionPanelView.swift` |
| Overlay + cursor | `clicky/ui/overlay.py`, `cursor.py` | `OverlayWindow.swift` |
| Listening pill | `clicky/ui/listening.py` | (new) |
| Colours, fonts, sizes | `clicky/ui/design.py` | (new) |
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

- Original idea, design, and macOS implementation: [**clicky.so**](https://clicky.so) by [@farzaa](https://github.com/farzaa/clicky). Go look at the original — it's the real deal.;
- This repo: a from-scratch Linux re-implementation in Python/PyQt6, written as a tribute so Linux users can run something close to the same experience.
- This fork builds on [msari06/clicky-linux](https://github.com/msari06/clicky-linux) by Mehmet Sari, which was itself built from Farza's Clicky.
## License

MIT (matches the original Clicky).
