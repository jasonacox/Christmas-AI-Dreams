# Release Notes

## v0.1.1 (unreleased)

- Improved splash UI with red & gold gradient, larger centered title
- Respect `REFRESH_SECONDS` env and helper script alignment
- Startup configuration logging with secrets redacted
- Graceful shutdown handling (SIGINT/SIGTERM) and FastAPI lifespan usage
- In-memory caching of last generated image so index serves it immediately
- `server.sh` for reliable Docker restart behavior and `PORT` support
- `upload.sh` tag confirmation and multi-arch build support (including arm/v7)
- README improvements and local development instructions

## v0.1.0 - Initial release

- Minimal FastAPI web service that generates festive AI images
- Supports `swarmui` and `openai` image providers
- Simple single-page UI that polls `/image` and displays generated scenes
- Configurable via environment variables (model, steps, size, refresh interval)

---

Notes:
- The top section lists recent improvements made in the working tree.
