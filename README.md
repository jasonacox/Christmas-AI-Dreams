# Christmas AI Dreaming

Are you dreaming about Christmas? This small web service asks an image model to imagine festive, fanciful Christmas scenes — from snowy village streets to cozy fireside gatherings — and displays the results in your browser. Prompts are generated automatically to produce varied, high-quality holiday imagery that refreshes at a configurable interval.

<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/f4f3d637-e569-42bb-aa47-c923247ce32e" style="width:400px;height:400px;object-fit:cover;border-radius:8px;box-shadow:0 6px 18px rgba(0,0,0,0.25);" />

The UI is intentionally simple: it maximizes each generated scene for maximum impact, shows the prompt used, and periodically requests a new image so the gallery keeps evolving like a holiday slideshow.

## Local development

Quick steps to run the project locally for development:

```bash
# clone the repo
git clone https://github.com/jasonacox/Christmas-AI-Dreams.git
cd Christmas-AI-Dreams

# create and activate a virtualenv (macOS / Linux)
python3 -m venv .venv
source .venv/bin/activate

# upgrade pip and install dependencies (the project ships chatbot/requirements.txt)
pip install --upgrade pip
if [ -f chatbot/requirements.txt ]; then
	pip install -r chatbot/requirements.txt
else
	pip install -r requirements.txt
fi

# set providers and run the server (example using local SwarmUI)
export IMAGE_PROVIDER=swarmui
export SWARMUI="http://10.0.1.25:7801"
export REFRESH_SECONDS=60
./run_local_server.sh

# or run directly
# python3 server.py
```

Notes:
- Use `IMAGE_PROVIDER=openai` and set `OPENAI_IMAGE_API_KEY` to use OpenAI image generation.
- Adjust `IMAGE_MODEL`, `IMAGE_STEPS`, `IMAGE_WIDTH`, and `REFRESH_SECONDS` via environment variables.

## Contributing

Ideas, fixes, and improvements are welcome. Fork the repo, make your changes, and open a pull request. If you have suggestions for new scene prompts, UI tweaks, or provider integrations, please contribute so the community can enjoy more festive AI dreams!

## Quick Start

The Christmas dream engine uses either a locally hosted SwarmUI installation or will use an OpenAI image-gen API compatible endpoint (e.g. dalle-e-3, gpt-image-1). To select the one you want use these simple docker commands to pull the engine and get it started. 

**SwarmUI Option**

```bash
docker run -p 4000:4000 \
	-e IMAGE_PROVIDER=swarmui \
	-e SWARMUI="http://10.0.1.25:7801" \
	-e IMAGE_MODEL="Flux/flux1-schnell-fp8" \
	-e IMAGE_CFGSCALE="1.0" \
	-e IMAGE_STEPS="6" \
	jasonacox/christmas-ai-dreams:latest
```

**OpenAI Option**

```bash
docker run -p 4000:4000 \
	-e IMAGE_PROVIDER=openai \
	-e OPENAI_IMAGE_API_KEY="sk-..." \
	-e OPENAI_IMAGE_API_BASE="https://api.openai.com/v1" \
	-e OPENAI_IMAGE_MODEL="dall-e-3" \
	-e OPENAI_IMAGE_SIZE="1024x1024" \
	jasonacox/christmas-ai-dreams:latest
```

## Local Development

## Share

Feel free to share any fun scenes in our Discussion forum.

<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/84170fc8-bf76-4cb2-b754-9ca8147ec45f" style="width:400px;height:400px;object-fit:cover;border-radius:8px;box-shadow:0 6px 18px rgba(0,0,0,0.25);" />

<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/623261c4-9d6b-4dc1-86a7-44b9eb6d2582" style="width:400px;height:400px;object-fit:cover;border-radius:8px;box-shadow:0 6px 18px rgba(0,0,0,0.25);" />

<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/83f2db5a-f2c7-41fb-9fa1-1f44cf32c527" style="width:400px;height:400px;object-fit:cover;border-radius:8px;box-shadow:0 6px 18px rgba(0,0,0,0.25);" />

<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/8cdfc88f-3a58-4a06-b12c-2b851320f115" style="width:400px;height:400px;object-fit:cover;border-radius:8px;box-shadow:0 6px 18px rgba(0,0,0,0.25);" />

<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/8105055a-6935-4467-a93a-a4699645b6bf" style="width:400px;height:400px;object-fit:cover;border-radius:8px;box-shadow:0 6px 18px rgba(0,0,0,0.25);" />

<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/c90e3f89-604d-470f-ae22-038f3f95844d" style="width:400px;height:400px;object-fit:cover;border-radius:8px;box-shadow:0 6px 18px rgba(0,0,0,0.25);" />

<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/bccecb03-58d6-4442-a988-2fd27f78808d" style="width:400px;height:400px;object-fit:cover;border-radius:8px;box-shadow:0 6px 18px rgba(0,0,0,0.25);" />

<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/c88c1533-3944-40f6-bd02-67a802a3fa73" style="width:400px;height:400px;object-fit:cover;border-radius:8px;box-shadow:0 6px 18px rgba(0,0,0,0.25);" />

<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/1ea366e0-836c-437b-bfe9-7189651b608f" style="width:400px;height:400px;object-fit:cover;border-radius:8px;box-shadow:0 6px 18px rgba(0,0,0,0.25);" />

<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/070f5de8-9ac1-418f-9713-8cc8cc59b1cb" style="width:400px;height:400px;object-fit:cover;border-radius:8px;box-shadow:0 6px 18px rgba(0,0,0,0.25);" />

