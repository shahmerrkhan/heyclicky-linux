.PHONY: help install run dev desktop-install desktop-uninstall check-system worker-install worker-dev worker-deploy clean

PYTHON ?= python3
VENV   ?= .venv

# Where desktop-install puts the app-menu and login entries (worked out at install time).
APPS_DIR      ?= $(or $(XDG_DATA_HOME),$(HOME)/.local/share)/applications
AUTOSTART_DIR ?= $(or $(XDG_CONFIG_HOME),$(HOME)/.config)/autostart
DESKTOP_MARK  := X-Clicky-Installed-By=make

help:
	@echo "Targets:"
	@echo "  make install         — Create a Python venv and install deps (via uv)"
	@echo "  make run             — Run the app"
	@echo "  make dev             — Run the app in dev mode (DEBUG logging)"
	@echo "  make desktop-install — Add Clicky to the app menu and start it at login"
	@echo "  make desktop-uninstall — Remove the app-menu and login entries"
	@echo "  make check-system    — Check Linux system dependencies"
	@echo "  make worker-install  — Install Cloudflare Worker deps (npm)"
	@echo "  make worker-dev      — Run the Worker locally on port 8787"
	@echo "  make worker-deploy   — Deploy the Worker to Cloudflare"
	@echo "  make clean           — Remove build artifacts"

install:
	@command -v uv >/dev/null 2>&1 || { echo "uv is not installed. See https://docs.astral.sh/uv/getting-started/installation/"; exit 1; }
	uv venv $(VENV)
	uv pip install --python $(VENV)/bin/python -e ".[dev]"
	@echo ""
	@echo "Install complete. To run: source $(VENV)/bin/activate && python -m clicky"

run:
	$(VENV)/bin/python -m clicky

dev:
	LOG_LEVEL=DEBUG $(VENV)/bin/python -m clicky

desktop-install:
	@[ -x "$(VENV)/bin/python" ] || { echo "No $(VENV) yet: run 'make install' first."; exit 1; }
	@case "$(CURDIR)" in *[\"\$$\\\`%\|]*) \
	  echo "The repo path has a character a launcher entry can't hold (\" \$$ \\ \` % |); move it."; exit 1;; esac
	@for f in "$(APPS_DIR)/clicky.desktop" "$(AUTOSTART_DIR)/clicky.desktop"; do \
	  if [ -e "$$f" ] && ! grep -qx '$(DESKTOP_MARK)' "$$f"; then \
	    echo "$$f exists and wasn't written by 'make desktop-install'; leaving it alone."; exit 1; fi; \
	done
	@chmod +x scripts/clicky-launch
	@mkdir -p "$(APPS_DIR)" "$(AUTOSTART_DIR)"
	@sed -e 's|@REPO@|$(CURDIR)|g' -e 's|@ARGS@||' scripts/clicky.desktop.in > "$(APPS_DIR)/clicky.desktop"
	@echo "App menu entry:  $(APPS_DIR)/clicky.desktop"
	@# Keep "off at login" if it was unticked in Session and Startup (XFCE writes Hidden=true).
	@hidden=; grep -qx 'Hidden=true' "$(AUTOSTART_DIR)/clicky.desktop" 2>/dev/null && hidden=1; \
	sed -e 's|@REPO@|$(CURDIR)|g' -e 's|@ARGS@| --delay 5|' scripts/clicky.desktop.in > "$(AUTOSTART_DIR)/clicky.desktop"; \
	if [ -n "$$hidden" ]; then echo 'Hidden=true' >> "$(AUTOSTART_DIR)/clicky.desktop"; \
	  echo "Login entry:     $(AUTOSTART_DIR)/clicky.desktop (kept switched off, as it was)"; \
	else echo "Login entry:     $(AUTOSTART_DIR)/clicky.desktop (starts 5 s after login)"; fi

desktop-uninstall:
	@for f in "$(APPS_DIR)/clicky.desktop" "$(AUTOSTART_DIR)/clicky.desktop"; do \
	  if [ ! -e "$$f" ]; then continue; fi; \
	  if grep -qx '$(DESKTOP_MARK)' "$$f"; then rm -f "$$f" && echo "Removed $$f"; \
	  else echo "$$f wasn't written by 'make desktop-install'; left alone."; fi; \
	done

check-system:
	@bash scripts/check-system.sh

worker-install:
	cd worker && npm install

worker-dev:
	cd worker && npx wrangler dev

worker-deploy:
	cd worker && npx wrangler deploy

clean:
	rm -rf $(VENV) build dist *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
