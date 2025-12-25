# Release Notes

## v0.1.3 - Optimized Icon Generation

- **Startup icon caching**: Favicon (multi-size ICO), apple-touch-icon (180×180 PNG), and 32×32 PNG favicon are now generated once at startup and stored in memory.
- **Simplified icon routes**: `/favicon.ico`, `/apple-touch-icon.png`, and `/favicon-32x32.png` now serve cached bytes directly (no per-request generation), returning 404 if cache unavailable.
- **Snowman favicon**: Dynamic snowman icon with hat and scarf for festive branding across browsers and iOS devices.
- Updated module docstring with comprehensive environment variable documentation and endpoint reference.

## v0.1.2 — Viewer-aware Generation

- Viewer tracking and generation gating: the server now tracks connected viewers and will pause image generation when no clients are connected (saves GPU cycles). Pages call `/connect` and `/disconnect` via `navigator.sendBeacon`.
- `/stats` endpoint: exposes `images_generated`, `current_connected`, `peak_connected`, and generation timing (min/max/avg) along with version/provider info.
- In-memory caching of the last generated image so reconnecting viewers see an image immediately.
- Expanded prompt vocabulary and `ALTERNATE_STYLES` (watercolor, storybook, vintage) to vary outputs and art styles.
- Measured generation timings and aggregated min/max/avg metrics for diagnostics.
- Improved Docker helpers: `server.sh` now detects host networking vs host.docker.internal and supports `PORT`; added a minimal `tiny.sh` helper to run the image with sane defaults.
- `upload.sh` enhancements: tag confirmation, multi-arch builds (including linux/arm/v7), and optional creation/push of an annotated git tag for releases.
- README and developer docs updated with clearer local development and run instructions.

## v0.1.1 - Minor Fixes

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
