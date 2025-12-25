#!/usr/bin/env python3
"""
Christmas AI Dreams - AI-generated Christmas scene web service.

Serves a single-page UI that polls for new AI-generated Christmas scenes every
REFRESH_SECONDS. Supports SwarmUI and OpenAI image providers. Features viewer
tracking (pauses generation when no viewers connected), image caching, dynamic
favicon generation at startup, and usage statistics.

Key Environment Variables:
  IMAGE_PROVIDER       - "swarmui" (default) or "openai"
  
  SwarmUI Settings:
    SWARMUI            - SwarmUI API base URL (default: http://localhost:7801)
    IMAGE_MODEL        - Model name (default: Flux/flux1-schnell-fp8)
    IMAGE_CFGSCALE     - CFG scale (default: 1.0)
    IMAGE_STEPS        - Generation steps (default: 6)
  
  OpenAI Settings:
    OPENAI_IMAGE_API_KEY    - OpenAI API key (required for openai provider)
    OPENAI_IMAGE_API_BASE   - API base URL (default: https://api.openai.com/v1)
    OPENAI_IMAGE_MODEL      - Model name (default: dall-e-3)
    OPENAI_IMAGE_SIZE       - Image size (default: 1024x1024)
  
  Server Settings:
    PORT               - HTTP port (default: 4000)
    REFRESH_SECONDS    - Client poll interval (default: 10)
    IMAGE_TIMEOUT      - Generation timeout in seconds (default: 300)
    APP_VERSION        - Override version string (default: v0.1.2)

Usage:
  # SwarmUI example
  export SWARMUI="http://10.0.1.25:7801"
  export IMAGE_MODEL="Flux/flux1-schnell-fp8"
  python3 server.py

  # OpenAI example
  export IMAGE_PROVIDER="openai"
  export OPENAI_IMAGE_API_KEY="sk-..."
  python3 server.py

Endpoints:
  /                  - Main UI (auto-refreshing image viewer)
  /image             - JSON: generates/returns new scene (or cached if viewers=0)
  /stats             - JSON: usage stats (images generated, viewers, timing)
  /favicon.ico       - Multi-size ICO favicon (cached at startup)
  /apple-touch-icon.png - 180x180 PNG for iOS (cached at startup)
  /favicon-32x32.png - 32x32 PNG favicon (cached at startup)
  /health            - Health check
  /version           - Version and provider info
  /connect           - POST: register viewer connection
  /disconnect        - POST: unregister viewer
  /viewers           - GET: current viewer count
"""
import io
import os
import sys
import random
import base64
import asyncio
import threading
import json
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
import uvicorn

# Ensure chatbot package is importable when running from repo root
ROOT = os.path.dirname(os.path.abspath(__file__))
CHATBOT_PATH = os.path.join(ROOT, "chatbot")
if CHATBOT_PATH not in sys.path:
    sys.path.insert(0, CHATBOT_PATH)

import aiohttp
from PIL import Image, ImageDraw

# Configuration (environment overrides)
PORT = int(os.environ.get("PORT", 4000))
SWARMUI = os.environ.get("SWARMUI", "http://localhost:7801")
IMAGE_MODEL = os.environ.get("IMAGE_MODEL", "Flux/flux1-schnell-fp8")
IMAGE_CFGSCALE = float(os.environ.get("IMAGE_CFGSCALE", 1.0))
IMAGE_STEPS = int(os.environ.get("IMAGE_STEPS", 6))
IMAGE_WIDTH = int(os.environ.get("IMAGE_WIDTH", 1024))
IMAGE_HEIGHT = int(os.environ.get("IMAGE_HEIGHT", 1024))
IMAGE_SEED = int(os.environ.get("IMAGE_SEED", -1))
IMAGE_TIMEOUT = int(os.environ.get("IMAGE_TIMEOUT", 300))
IMAGE_PROVIDER = os.environ.get("IMAGE_PROVIDER", "swarmui").lower()

# Server version (can be overridden with APP_VERSION env)
VERSION = os.environ.get("APP_VERSION", "v0.1.3")

# OpenAI image settings
OPENAI_IMAGE_API_KEY = os.environ.get("OPENAI_IMAGE_API_KEY", "")
OPENAI_IMAGE_API_BASE = os.environ.get("OPENAI_IMAGE_API_BASE", "https://api.openai.com/v1")
OPENAI_IMAGE_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "dall-e-3")
OPENAI_IMAGE_SIZE = os.environ.get("OPENAI_IMAGE_SIZE", "1024x1024")

# Default refresh interval in seconds (front-end will poll /image)
# Read `REFRESH_SECONDS` env var (fallback to 10s).
DEFAULT_REFRESH = int(os.environ.get("REFRESH_SECONDS", "10"))

app = FastAPI()

# In-memory cache of the last generated image + lock for thread-safety
LAST_IMAGE: dict | None = None
LAST_IMAGE_LOCK = threading.Lock()
# In-memory cached icons (generated on startup)
ICON_LOCK = threading.Lock()
APPLE_TOUCH_BYTES: bytes | None = None
FAVICON_32_BYTES: bytes | None = None
FAVICON_ICO_BYTES: bytes | None = None
# Track connected viewers (increment on page load, decrement on unload)
CONNECTED_VIEWERS = 0
VIEWERS_LOCK = threading.Lock()
# Stats
IMAGES_GENERATED = 0
MAX_CONNECTED_VIEWERS = 0
STATS_LOCK = threading.Lock()
# Generation time stats (seconds)
GEN_TIME_COUNT = 0
GEN_TIME_SUM = 0.0
GEN_TIME_MIN: float | None = None
GEN_TIME_MAX: float | None = None

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("christmas_server")
logger.info("Starting Christmas Scenes server - version %s", VERSION)
# Log effective configuration on startup (redact sensitive values)
try:
    config = {
        "PORT": PORT,
        "SWARMUI": SWARMUI,
        "IMAGE_PROVIDER": IMAGE_PROVIDER,
        "IMAGE_MODEL": IMAGE_MODEL,
        "IMAGE_CFGSCALE": IMAGE_CFGSCALE,
        "IMAGE_STEPS": IMAGE_STEPS,
        "IMAGE_WIDTH": IMAGE_WIDTH,
        "IMAGE_HEIGHT": IMAGE_HEIGHT,
        "IMAGE_SEED": IMAGE_SEED,
        "IMAGE_TIMEOUT": IMAGE_TIMEOUT,
        "REFRESH_SECONDS": DEFAULT_REFRESH,
        "OPENAI_IMAGE_API_BASE": OPENAI_IMAGE_API_BASE,
        "OPENAI_IMAGE_MODEL": OPENAI_IMAGE_MODEL,
        "OPENAI_IMAGE_SIZE": OPENAI_IMAGE_SIZE,
        # Do not log secret values — only indicate presence
        "OPENAI_IMAGE_API_KEY": ("SET" if OPENAI_IMAGE_API_KEY else "NOT SET"),
    }
    logger.info("Effective configuration:\n%s", "\n".join(f"{k}: {v}" for k, v in config.items()))
except Exception:
    logger.exception("Failed to log configuration")


@asynccontextmanager
async def _lifespan(app):
    logger.info("Application startup — generating cached assets and ready to serve requests.")
    # Generate and cache small PNG icons to avoid regenerating on each request
    try:
        def _make_snowman_image(size: int) -> Image.Image:
            im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            draw = ImageDraw.Draw(im)
            cx = size // 2
            bottom_r = int(size * 0.25)
            mid_r = int(size * 0.17)
            head_r = int(size * 0.11)
            bottom_cy = size - (size // 8) - bottom_r
            mid_cy = bottom_cy - bottom_r - mid_r + int(size * 0.05)
            head_cy = mid_cy - mid_r - head_r + int(size * 0.03)

            def circ(x, y, r, fill=(255, 255, 255, 255), outline=(0, 0, 0, 255)):
                draw.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline=outline)

            circ(cx, bottom_cy, bottom_r)
            circ(cx, mid_cy, mid_r)
            circ(cx, head_cy, head_r)
            eye_r = max(1, int(size * 0.015))
            draw.ellipse((cx - int(size * 0.05) - eye_r, head_cy - int(size * 0.03) - eye_r, cx - int(size * 0.05) + eye_r, head_cy - int(size * 0.03) + eye_r), fill=(0, 0, 0, 255))
            draw.ellipse((cx + int(size * 0.05) - eye_r, head_cy - int(size * 0.03) - eye_r, cx + int(size * 0.05) + eye_r, head_cy - int(size * 0.03) + eye_r), fill=(0, 0, 0, 255))
            nose = [(cx, head_cy), (cx + int(size * 0.11), head_cy - int(size * 0.05)), (cx + int(size * 0.11), head_cy + int(size * 0.05))]
            draw.polygon(nose, fill=(255, 140, 0, 255))
            hat_w = head_r * 2 + int(size * 0.08)
            brim_top = head_cy - head_r - int(size * 0.08)
            brim_h = max(1, int(size * 0.05))
            draw.rectangle((cx - hat_w // 2, brim_top, cx + hat_w // 2, brim_top + brim_h), fill=(20, 20, 20, 255))
            draw.rectangle((cx - (hat_w // 2 - int(size * 0.04)), brim_top - int(size * 0.16), cx + (hat_w // 2 - int(size * 0.04)), brim_top), fill=(30, 30, 30, 255))
            scarf_top = mid_cy - int(size * 0.12)
            draw.rectangle((cx - mid_r, scarf_top, cx + mid_r, scarf_top + int(size * 0.08)), fill=(200, 30, 30, 255))
            return im

        def _png_bytes_from_image(img: Image.Image) -> bytes:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()

        # Build and cache the common sizes (PNG + multi-size ICO)
        global APPLE_TOUCH_BYTES, FAVICON_32_BYTES, FAVICON_ICO_BYTES
        with ICON_LOCK:
            try:
                if APPLE_TOUCH_BYTES is None:
                    APPLE_TOUCH_BYTES = _png_bytes_from_image(_make_snowman_image(180))
                if FAVICON_32_BYTES is None:
                    FAVICON_32_BYTES = _png_bytes_from_image(_make_snowman_image(32))
                if FAVICON_ICO_BYTES is None:
                    # Create ICO containing several sizes
                    sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
                    max_size = max(s[0] for s in sizes)
                    base = _make_snowman_image(max_size)
                    buf = io.BytesIO()
                    base.save(buf, format="ICO", sizes=sizes)
                    FAVICON_ICO_BYTES = buf.getvalue()
            except Exception:
                logger.exception("Failed to generate cached icons at startup")
    except Exception:
        logger.exception("Unexpected error during startup icon generation")
    try:
        yield
    finally:
        logger.info("Application shutdown initiated — performing cleanup.")

# Use the lifespan context to avoid deprecated on_event handlers
app.router.lifespan_context = _lifespan

# Simple list of scene ideas
SCENE_KEYWORDS = [
    "winter snow scene",
    "cozy warm fireplace scene",
    "Christmas tree with lights",
    "decorations and lights on a house",
    "pile of Christmas presents",
    "family Christmas dinner table",
    "nativity scene",
    "Bethlehem star over a town",
    "Santa Claus meeting children",
    "Santa's workshop with elves",
    "Santa's sleigh in the night sky",
    "reindeer in a snowy field",
    "ice skating on a frozen pond",
    "children building a snowman",
    "carolers singing in the snow",
    "Christmas wreath on a door",
    "hot cocoa by the fireplace",
    "Christmas cookies on a plate",
    "snow-covered pine forest",
    "festive holiday village",
    "Christmas lights on a tree at night",
    "winter snow-covered cottage at dusk",
    "holiday market with wooden stalls and twinkling lights",
    "enchanted northern-lights over a pine forest",
    "ice castle with frosted turrets",
    "cozy kitchen baking cookies",
    "Victorian street with vintage decorations",
    "toy-train circling a decorated tree",
    "snow globe miniature village",
    "rooftop silhouette with sleigh in the sky",
]

STYLE_PREFIX = (
    "Ultra-detailed, cinematic, photorealistic, 8k, dramatic lighting, "
    "warm color grading, high dynamic range, shallow depth of field"
)

def build_prompt() -> str:
    """Return a randomized Christmas scene prompt."""
    scene = random.choice(SCENE_KEYWORDS)
    extras = [
        "snow falling softly",
        "warm glow from lanterns",
        "children playing",
        "candles and garlands",
        "cozy wool textures",
        "gold and red ornaments",
        "soft bokeh lights",
        "steam rising from mugs of hot cocoa",
        "frosted window patterns",
        "gingerbread textures and icing",
        "elves wrapping gifts",
        "gentle film grain",
        "reflections on wet cobblestone",
    ]
    take = random.sample(extras, k=2)

    # Occasionally use an alternate illustrative style for variety
    ALTERNATE_STYLES = [
        "Whimsical, storybook illustration, watercolor, soft palette, hand-painted",
        "Vintage postcard, warm tones, slight film grain, nostalgic",
        "Painterly, oil painting, soft brush strokes, cozy mood",
        "Children's book illustration, flat colors, high charm",
    ]
    style_prefix = STYLE_PREFIX
    if random.random() < 0.2:  # 20% chance to choose an alternate style
        style_prefix = random.choice(ALTERNATE_STYLES)

    prompt = f"{style_prefix}, {scene}, {take[0]}, {take[1]}, festive atmosphere"
    logger.info("Built prompt: %s", prompt)
    return prompt


async def generate_scene(prompt: str | None = None) -> dict:
    """Generate an image using the SwarmUI generator and return a data URI and metadata."""
    if prompt is None:
        prompt = build_prompt()

    logger.info("Image provider: %s", IMAGE_PROVIDER)
    if IMAGE_PROVIDER == "swarmui":
        logger.info("Sending prompt to SwarmUI (%s) model=%s", SWARMUI, IMAGE_MODEL)
        logger.info("Prompt: %s", prompt)

        # Use a lightweight internal SwarmUI client to avoid depending on chatbot package
        async def _get_session_id(session: aiohttp.ClientSession) -> str | None:
            try:
                async with session.post(f"{SWARMUI.rstrip('/')}/API/GetNewSession", json={}, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("session_id")
            except Exception as e:
                logger.error("Error getting session id from SwarmUI: %s", e)
            return None

        async def _call_generate(session: aiohttp.ClientSession, session_id: str, prompt_text: str) -> str | None:
            params = {
                "model": IMAGE_MODEL,
                "width": IMAGE_WIDTH,
                "height": IMAGE_HEIGHT,
                "cfgscale": IMAGE_CFGSCALE,
                "steps": IMAGE_STEPS,
                "seed": IMAGE_SEED,
            }
            raw_input = {"prompt": str(prompt_text), **{k: v for k, v in params.items()}, "donotsave": True}
            data = {
                "session_id": session_id,
                "images": "1",
                "prompt": str(prompt_text),
                **{k: str(v) for k, v in params.items()},
                "donotsave": True,
                "rawInput": raw_input,
            }
            try:
                async with session.post(f"{SWARMUI.rstrip('/')}/API/GenerateText2Image", json=data, timeout=IMAGE_TIMEOUT) as resp:
                    if resp.status == 200:
                        j = await resp.json()
                        imgs = j.get("images") or []
                        if imgs:
                            return imgs[0]
                    else:
                        logger.error("SwarmUI GenerateText2Image returned status %s", resp.status)
            except Exception as e:
                logger.error("Error calling SwarmUI GenerateText2Image: %s", e)
            return None

        image_encoded = None
        try:
            async with aiohttp.ClientSession() as session:
                session_id = await _get_session_id(session)
                if not session_id:
                    logger.error("Unable to obtain SwarmUI session id")
                    return {"error": "No session"}
                image_encoded = await _call_generate(session, session_id, prompt)
        except Exception as e:
            logger.error("Unexpected error during SwarmUI generation: %s", e)
            return {"error": "Generation exception"}

        if not image_encoded:
            logger.error("Image generation failed for prompt: %s", prompt)
            return {"error": "Generation failed"}
    elif IMAGE_PROVIDER == "openai":
        logger.info("Sending prompt to OpenAI Images API (%s) model=%s", OPENAI_IMAGE_API_BASE, OPENAI_IMAGE_MODEL)
        logger.info("Prompt: %s", prompt)

        async def _call_openai(session: aiohttp.ClientSession, prompt_text: str) -> str | None:
            url = f"{OPENAI_IMAGE_API_BASE.rstrip('/')}/images/generations"
            headers = {"Authorization": f"Bearer {OPENAI_IMAGE_API_KEY}", "Content-Type": "application/json"}
            body = {"model": OPENAI_IMAGE_MODEL, "prompt": prompt_text, "size": OPENAI_IMAGE_SIZE}
            try:
                async with session.post(url, json=body, headers=headers, timeout=IMAGE_TIMEOUT) as resp:
                    if resp.status == 200:
                        j = await resp.json()
                        # Support both b64_json and url returns
                        data = j.get("data") or []
                        if data:
                            first = data[0]
                            if "b64_json" in first:
                                return first["b64_json"]
                            if "url" in first:
                                # fetch binary and return as base64
                                img_url = first["url"]
                                async with session.get(img_url) as img_resp:
                                    if img_resp.status == 200:
                                        b = await img_resp.read()
                                        return base64.b64encode(b).decode("utf-8")
                    else:
                        text = await resp.text()
                        logger.error("OpenAI images API returned %s: %s", resp.status, text)
            except Exception as e:
                logger.error("Error calling OpenAI Images API: %s", e)
            return None

        image_encoded = None
        try:
            async with aiohttp.ClientSession() as session:
                image_encoded = await _call_openai(session, prompt)
        except Exception as e:
            logger.error("Unexpected error during OpenAI generation: %s", e)
            return {"error": "Generation exception"}

        if not image_encoded:
            logger.error("OpenAI image generation failed for prompt: %s", prompt)
            return {"error": "Generation failed"}
    else:
        logger.error("Unknown IMAGE_PROVIDER: %s", IMAGE_PROVIDER)
        return {"error": "Unsupported image provider"}

    # Normalize to raw base64 payload
    if "," in image_encoded:
        image_b64 = image_encoded.split(",", 1)[1]
    else:
        image_b64 = image_encoded

    logger.info("Received image data (bytes ~ %d)", len(image_b64))

    try:
        image = Image.open(io.BytesIO(base64.b64decode(image_b64)))
    except Exception:
        return {"error": "Unable to decode image data"}

    # Resize down for web if necessary
    max_dim = 1024
    if image.width > max_dim or image.height > max_dim:
        image.thumbnail((max_dim, max_dim))
    # Convert to JPEG for browser-friendliness
    if image.mode == "RGBA":
        image = image.convert("RGB")
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=90)
    out_b64 = base64.b64encode(out.getvalue()).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{out_b64}"

    return {"prompt": prompt, "image_data": data_uri}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, refresh: int | None = None):
    """Serve a minimal HTML page that polls `/image` every X seconds."""
    interval = refresh or DEFAULT_REFRESH
    # If we have a cached last image, embed it so the page shows immediately
    try:
        with LAST_IMAGE_LOCK:
            cached = LAST_IMAGE
    except Exception:
        cached = None

    initial_image_js = json.dumps(cached.get("image_data")) if cached else "null"
    initial_prompt_js = json.dumps(cached.get("prompt")) if cached else "null"

    html = f"""
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8" />
                <link rel="icon" href="/favicon.ico" />
                <link rel="apple-touch-icon" href="/apple-touch-icon.png" />
                <meta name="theme-color" content="#b30000" />
        <meta name="viewport" content="width=device-width,initial-scale=1" />
        <title>Christmas Scenes</title>
        <style>
            html,body {{ height:100%; margin:0; background:#111; color:#fff; display:flex; align-items:center; justify-content:center; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
            #img {{ max-width:100%; max-height:100vh; box-shadow: 0 8px 30px rgba(0,0,0,0.6); }}
            /* Bottom-center translucent prompt overlay */
            #meta {{ position:fixed; left:50%; bottom:8px; transform:translateX(-50%); background:rgba(0,0,0,0.25); padding:4px 6px; border-radius:6px; font-family:Helvetica,Arial; font-size:12px; opacity:0.5; color:#fff; text-align:center; pointer-events:none; max-width:90%; }}
            #prompt {{ font-size:0.9em; }}
            /* Modern splash screen styling (red & gold theme) */
            #splash {{ display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; padding:40px; background:linear-gradient(135deg, rgba(178,17,17,0.18), rgba(255,215,0,0.12)); border-radius:20px; box-shadow:0 20px 60px rgba(0,0,0,0.5); max-width:600px; }}
            #splash-text {{ font-size:3.5em; font-weight:700; margin-bottom:20px; background:linear-gradient(45deg, #b30000, #ffd700); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; letter-spacing:2px; text-shadow:2px 2px 6px rgba(0,0,0,0.4); }}
            #splash-link {{ margin-top:15px; font-size:1em; opacity:0.95; }}
            #splash-link a {{ color:#ffd700; text-decoration:none; transition:all 0.25s ease; }}
            #splash-link a:hover {{ color:#fff; text-shadow:0 1px 0 rgba(0,0,0,0.6); }}
            #splash-version {{ margin-top:25px; font-size:0.9em; opacity:0.6; font-weight:300; }}
        </style>
      </head>
            <body>
                <img id="img" src="" alt="Christmas scene" style="display:none;" />
                <div id="splash">
                        <div id="splash-text">Christmas AI Dreaming</div>
                        <div id="splash-link"><a href="http://github.com/jasonacox/Christmas-AI-Dreams" target="_blank" rel="noopener">github.com/jasonacox/Christmas-AI-Dreams</a></div>
                        <div id="splash-version">Version: {VERSION}</div>
                </div>
                <div id="meta">Prompt: <span id="prompt">(generating) - Please Wait...</span></div>
                <script>
                    const interval = {interval} * 1000;
                    const initialImage = {initial_image_js};
                    const initialPrompt = {initial_prompt_js};
                    // Show splash immediately while image generation may be in progress
                    const splash = document.getElementById('splash');
                    const img = document.getElementById('img');
                    const promptEl = document.getElementById('prompt');

                    // Notify server we're connected (use sendBeacon for unload-safe POST)
                    try {{
                        navigator.sendBeacon('/connect');
                    }} catch (e) {{ /* ignore */ }}

                    // If a cached image exists, show it immediately and hide splash
                    if (initialImage) {{
                        img.src = initialImage;
                        promptEl.textContent = initialPrompt || '';
                        img.style.display = '';
                        splash.style.display = 'none';
                    }}

                    // Notify server on unload that we're disconnecting
                    window.addEventListener('beforeunload', function() {{
                        try {{ navigator.sendBeacon('/disconnect'); }} catch (e) {{}}
                    }});

                    async function fetchImage() {{
                        try {{
                            const res = await fetch('/image');
                            if (!res.ok) return;
                            const j = await res.json();
                            if (j.image_data) {{
                                // Set image and hide splash
                                img.src = j.image_data;
                                promptEl.textContent = j.prompt || '';
                                img.style.display = '';
                                splash.style.display = 'none';
                            }} else {{
                                // keep showing splash
                                splash.style.display = '';
                            }}
                        }} catch (e) {{
                            console.error(e);
                        }}
                    }}
                    // Fetch in background immediately, then poll
                    fetchImage();
                    setInterval(fetchImage, interval);
                </script>
            </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/image")
async def image_endpoint(prompt: str | None = None):
    """Generate and return a new Christmas scene as JSON with a `image_data` data URI."""
    # If no prompt override and there are zero connected viewers, skip generation
    if prompt is None:
        try:
            with VIEWERS_LOCK:
                viewers = CONNECTED_VIEWERS
        except Exception:
            viewers = 0
        if viewers == 0:
            logger.info("No connected viewers detected; generation paused until viewers connect")
            return JSONResponse(status_code=429, content={"error": "No connected viewers — generation paused"})

    # Measure generation time and update stats only for successful generations
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    result = await generate_scene(prompt)
    elapsed = loop.time() - t0
    if "error" not in result:
        try:
            with STATS_LOCK:
                global IMAGES_GENERATED, GEN_TIME_COUNT, GEN_TIME_SUM, GEN_TIME_MIN, GEN_TIME_MAX
                IMAGES_GENERATED += 1
                GEN_TIME_COUNT += 1
                GEN_TIME_SUM += elapsed
                if GEN_TIME_MIN is None or elapsed < GEN_TIME_MIN:
                    GEN_TIME_MIN = elapsed
                if GEN_TIME_MAX is None or elapsed > GEN_TIME_MAX:
                    GEN_TIME_MAX = elapsed
        except Exception:
            logger.exception("Failed to update generation stats")
    if "error" in result:
        return JSONResponse(status_code=500, content={"error": result["error"]})
    # Cache the last successful image so the index page can show it immediately
    try:
        with LAST_IMAGE_LOCK:
            global LAST_IMAGE
            LAST_IMAGE = result
    except Exception:
        logger.exception("Failed to cache last image")
    return JSONResponse(content=result)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/version")
async def version():
    """Return service version and image provider details."""
    data = {
        "version": VERSION,
        "image_provider": IMAGE_PROVIDER,
    }
    try:
        if IMAGE_PROVIDER == "swarmui":
            data.update({"swarmui": SWARMUI, "model": IMAGE_MODEL})
        elif IMAGE_PROVIDER == "openai":
            data.update({"openai_base": OPENAI_IMAGE_API_BASE, "model": OPENAI_IMAGE_MODEL})
    except Exception:
        pass
    return data


@app.get("/stats")
async def stats():
    """Return usage and generation statistics."""
    try:
        with VIEWERS_LOCK:
            current = CONNECTED_VIEWERS
            peak = MAX_CONNECTED_VIEWERS
        with STATS_LOCK:
            count = IMAGES_GENERATED
            gen_count = GEN_TIME_COUNT
            gen_sum = GEN_TIME_SUM
            gen_min = GEN_TIME_MIN
            gen_max = GEN_TIME_MAX
        avg = (gen_sum / gen_count) if gen_count > 0 else None
        return {
            "version": VERSION,
            "image_provider": IMAGE_PROVIDER,
            "current_connected": current,
            "peak_connected": peak,
            "images_generated": count,
            "generation_time_min_s": gen_min,
            "generation_time_max_s": gen_max,
            "generation_time_avg_s": avg,
        }
    except Exception:
        logger.exception("Failed to read stats")
        return JSONResponse(status_code=500, content={"error": "unable to read stats"})


@app.get("/favicon.ico")
async def favicon():
    """Return cached multi-size ICO favicon."""
    try:
        with ICON_LOCK:
            if FAVICON_ICO_BYTES:
                headers = {"Cache-Control": "public, max-age=86400"}
                return Response(content=FAVICON_ICO_BYTES, media_type="image/x-icon", headers=headers)
        logger.error("Favicon ICO cache is empty")
        return JSONResponse(status_code=404, content={"error": "favicon not available"})
    except Exception:
        logger.exception("Failed to serve favicon")
        return JSONResponse(status_code=500, content={"error": "favicon serve failed"})


@app.get("/apple-touch-icon.png")
async def apple_touch_icon():
    """Return cached 180x180 PNG for Apple touch icons."""
    try:
        with ICON_LOCK:
            if APPLE_TOUCH_BYTES:
                headers = {"Cache-Control": "public, max-age=86400"}
                return Response(content=APPLE_TOUCH_BYTES, media_type="image/png", headers=headers)
        logger.error("Apple touch icon cache is empty")
        return JSONResponse(status_code=404, content={"error": "apple icon not available"})
    except Exception:
        logger.exception("Failed to serve apple-touch-icon")
        return JSONResponse(status_code=500, content={"error": "apple icon serve failed"})


@app.get("/favicon-32x32.png")
async def favicon_32():
    """Return cached 32x32 PNG favicon."""
    try:
        with ICON_LOCK:
            if FAVICON_32_BYTES:
                headers = {"Cache-Control": "public, max-age=86400"}
                return Response(content=FAVICON_32_BYTES, media_type="image/png", headers=headers)
        logger.error("32x32 favicon cache is empty")
        return JSONResponse(status_code=404, content={"error": "favicon not available"})
    except Exception:
        logger.exception("Failed to serve favicon-32x32")
        return JSONResponse(status_code=500, content={"error": "favicon serve failed"})


@app.post("/connect")
async def connect(request: Request):
    """Mark a viewer as connected. Called from the page via `navigator.sendBeacon`."""
    try:
        with VIEWERS_LOCK:
            global CONNECTED_VIEWERS, MAX_CONNECTED_VIEWERS
            CONNECTED_VIEWERS += 1
            current = CONNECTED_VIEWERS
            if current > MAX_CONNECTED_VIEWERS:
                MAX_CONNECTED_VIEWERS = current
        logger.info("Viewer connected — total=%d (peak=%d)", current, MAX_CONNECTED_VIEWERS)
        return {"connected": current}
    except Exception:
        logger.exception("Failed to register connect")
        return JSONResponse(status_code=500, content={"error": "connect failed"})


@app.post("/disconnect")
async def disconnect(request: Request):
    """Mark a viewer as disconnected. Called from the page via `navigator.sendBeacon`."""
    try:
        with VIEWERS_LOCK:
            global CONNECTED_VIEWERS
            if CONNECTED_VIEWERS > 0:
                CONNECTED_VIEWERS -= 1
            current = CONNECTED_VIEWERS
        logger.info("Viewer disconnected — total=%d", current)
        return {"connected": current}
    except Exception:
        logger.exception("Failed to register disconnect")
        return JSONResponse(status_code=500, content={"error": "disconnect failed"})


@app.get("/viewers")
async def viewers():
    """Return current viewer count."""
    try:
        with VIEWERS_LOCK:
            current = CONNECTED_VIEWERS
        return {"connected": current}
    except Exception:
        logger.exception("Failed to read viewers")
        return JSONResponse(status_code=500, content={"error": "unable to read viewers"})


if __name__ == '__main__':
    # Optional quick connectivity check to SwarmUI
    print(f"Starting Christmas Scenes server on port {PORT}")
    print(f"SwarmUI host: {SWARMUI} model: {IMAGE_MODEL}")
    # Run uvicorn programmatically and install our own signal handlers so
    # shutdown can be handled gracefully (useful for Ctrl-C and Docker SIGTERM).
    import signal

    config = uvicorn.Config(app, host='0.0.0.0', port=PORT, log_level="info", loop="asyncio", lifespan="on")
    server = uvicorn.Server(config)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _handle_signal(sig_num, frame):
        try:
            name = signal.Signals(sig_num).name
        except Exception:
            name = str(sig_num)
        logger.info("Received signal %s, initiating graceful shutdown...", name)
        # Ask the server to shutdown asynchronously
        loop.create_task(server.shutdown())

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        loop.run_until_complete(server.serve())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Interrupted — shutting down")
    except Exception:
        logger.exception("Server error")
    finally:
        # Ensure loop is cleanly closed
        try:
            pending = asyncio.all_tasks(loop=loop)
            for t in pending:
                t.cancel()
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            pass
        loop.close()
