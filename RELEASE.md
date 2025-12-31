# Release Notes

## v0.1.4 - Session Tracking and Code Quality

- **Session-based tracking**: Each viewer is tracked by a unique session ID (from `X-Session-ID` header or generated from IP+User-Agent), replacing simple increment/decrement counters.
- **5-minute session TTL**: Sessions expire after 5 minutes of inactivity; background cleanup task runs every minute to remove stale sessions.
- **LRU session eviction**: Sessions now use `OrderedDict` with a maximum limit of 1000 sessions, evicting oldest when limit is reached.
- **Auto-refresh on activity**: `/image` requests automatically register/refresh the session, keeping downloads counted as active sessions.
- **Image Caching**: Image generation respects `REFRESH_SECONDS` — subsequent requests within the interval return the cached image instead of generating a new one.
- **Enhanced `/stats`**: Now includes `active_sessions`, `session_ttl_s`, `last_activity_ts`, `last_activity_age_s`, `last_image_ts`, `last_image_age_s`, and icon cache status flags (removed locks for better performance).
- **Code quality improvements**:
  - Module-level constants for magic numbers (icon sizes, cache durations, session limits)
  - Consolidated thread locks (SESSION_STATE_LOCK, IMAGE_CACHE_LOCK, STATS_LOCK)
  - Removed unnecessary locks from read-only icon endpoints and informational stats
  - Extracted provider-specific logic into `_generate_swarmui()` and `_generate_openai()` functions
  - Removed prompt parameter from `/image` endpoint (server-only random generation)
  - Added provider name to generation log messages for better debugging

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
