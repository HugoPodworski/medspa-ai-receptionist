# Medspa AI Receptionist (YouTube Edition)

This repository accompanies my YouTube walkthrough of building and deploying a voice AI receptionist. It’s optimized for viewers following along step-by-step. After the first deployment, any scaling or resource changes should be done by editing `fly.toml` and redeploying—no code changes needed.

---

## Local Development (Run It On Your Machine)

### Prerequisites
- **Python**: 3.10–3.12
- **uv**: Python package/deps manager
- **ngrok**: To expose your local server to the internet
- **Accounts/keys**: Daily API key, Twilio account + phone number, Deepgram, Cartesia, Cerebras, and Qdrant (optional for RAG)

Install tools on macOS:
```bash
# uv
brew install uv

# ngrok
brew install ngrok
# or download from https://ngrok.com/download
```

### 1) Clone the repo
```bash
git clone https://github.com/HugoPodworski/medspa-ai-receptionist.git
cd medspa-ai-receptionist
```

### 2) Create virtual environment and install deps
```bash
# Create venv (recommended)
uv venv
source .venv/bin/activate

# Install dependencies from pyproject.toml/uv.lock
uv sync
```

### 3) Configure environment variables
```bash
# Start from the template
cp env.example .env
# Open .env and fill in your own keys
```
Required values (see `env.example`):
- DAILY_API_KEY (and optional DAILY_API_URL)
- TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN
- DEEPGRAM_API_KEY, CARTESIA_API_KEY, CEREBRAS_API_KEY
- QDRANT_URL, QDRANT_API_KEY, RAG_COLLECTION_NAME (RAG optional; app degrades gracefully)
- ENVIRONMENT=local (for local development)

### 4) Run the server locally
Pick one of the following:
```bash
# Easiest: run the script directly (reload via __main__)
uv run server.py

# Or run uvicorn explicitly
uv run uvicorn server:app --host 0.0.0.0 --port 7860 --reload
```
Verify it’s up:
```bash
curl http://localhost:7860/health
# => {"status":"healthy"}
```

### 5) Expose your server with ngrok
```bash
# Authenticate once (from your ngrok dashboard)
ngrok config add-authtoken <YOUR_NGROK_AUTHTOKEN>

# Start a tunnel to your local port
ngrok http 7860
```
Copy the public URL shown by ngrok, e.g., `https://random-subdomain.ngrok.io`.

### 6) Point Twilio to your local server
In Twilio Console:
1. Go to Phone Numbers → Manage → Active numbers → select your number
2. Under Voice configuration, set:
   - A Call Comes In: Webhook `https://<your-ngrok-domain>/call`
   - HTTP Method: POST
3. Save

### 7) Test end-to-end locally
- Call your Twilio phone number
- You should hear: “Thank you for calling Thérapie Clinic, how can I help you today?”
- Watch local logs in your terminal and ngrok console

Troubleshooting tips:
- If you see 403 errors from Qdrant or other services, double-check API keys in `.env`
- If /health works locally but Twilio fails, ensure ngrok is running and the webhook URL is correct (`/call`, POST)
- If you change `.env`, restart the server

---

# Complete Step-by-Step First Deployment to Fly.io

## Prerequisites
```bash
# Install Fly CLI
brew install flyctl  # macOS
# or follow: https://fly.io/docs/getting-started/installing-flyctl/

# Create account and login
fly auth signup  # or fly auth login
```

## 1. Project Configuration Files

### Create `fly.toml` in project root:
```toml
app = 'your-unique-app-name'  # Change this!
primary_region = 'iad'  # or your preferred region

[build]

[env]
  PORT = "7860"
  ENVIRONMENT = "local"

[http_service]
  internal_port = 7860
  force_https = true
  auto_stop_machines = 'stop'
  auto_start_machines = true
  min_machines_running = 0
  processes = ['app']

  [[http_service.checks]]
    grace_period = "60s"
    interval = "30s"
    method = "get"
    path = "/health"
    protocol = "http"
    timeout = "30s"

[[vm]]
  memory = '2048'  # Start with 2GB, scale up if needed
  cpu_kind = 'shared'
  cpus = 1
```

### Verify your `Dockerfile`:
```dockerfile
FROM python:3.12-slim

# Do not write .pyc files to reduce image size
ENV PYTHONDONTWRITEBYTECODE=1
# Prefer CPU-only PyTorch wheels
ENV UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
# Allow uv to choose best versions across all indexes
ENV UV_INDEX_STRATEGY=unsafe-best-match

# Install uv inside the container (multi-arch friendly)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy lockfile and project config, then install deps
COPY ./uv.lock uv.lock
COPY ./pyproject.toml pyproject.toml
RUN uv sync --no-install-project --no-dev \
    && rm -rf /root/.cache /root/.local/share/uv

# Copy the application code
COPY ./bot.py bot.py
COPY ./model_config.py model_config.py
COPY ./ragprocessing.py ragprocessing.py
COPY ./server.py server.py

# Expose FastAPI port
EXPOSE 7860

# Default command runs via uv to ensure deps/env are applied
CMD ["uv", "run", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
```

### Update `pyproject.toml` (remove unused extras):
```toml
[project]
name = "daily-twilio-sip-dial-in"
version = "0.1.0" 
description = "Daily SIP + Twilio Dial-in example"
requires-python = ">=3.10,<3.13"
dependencies = [
    "pipecat-ai[daily,deepgram,cartesia,openai,silero,runner,remote-smart-turn]>=0.0.83",
    "pipecatcloud>=0.2.4",
    "twilio==9.8.0",
    "sentence-transformers==5.1.0",
    "qdrant-client==1.15.1", 
    "python-dotenv>=1.0.1,<2.0.0",
]
```

## 2. Create Fly App

```bash
# From your project directory
fly launch --no-deploy
```

During launch:
- App name: Choose something unique (e.g., `your-name-voice-bot`)
- Region: Choose closest to your users (e.g., `iad` for US East)
- Don't deploy yet: We need to set secrets first

## 3. Set Environment Secrets

```bash
fly secrets set \
  DAILY_API_KEY=your_daily_api_key \
  TWILIO_ACCOUNT_SID=your_twilio_account_sid \
  TWILIO_AUTH_TOKEN=your_twilio_auth_token \
  DEEPGRAM_API_KEY=your_deepgram_api_key \
  CARTESIA_API_KEY=your_cartesia_api_key \
  CEREBRAS_API_KEY=your_cerebras_api_key \
  FAL_SMART_TURN_API_KEY=your_fal_smart_turn_api_key \
  QDRANT_URL=your_qdrant_url \
  QDRANT_API_KEY=your_qdrant_api_key \
  RAG_COLLECTION_NAME=therapie_clinic_rag \
  ENVIRONMENT=local
```

## 4. First Deployment

```bash
# Deploy to Fly
fly deploy --remote-only
```

After this first deploy: to change CPUs/memory, update `fly.toml` and redeploy. No code changes needed for scaling.

What happens during deployment:
- Builds Docker image (takes ~5-10 minutes first time)
- Pushes image to Fly registry
- Creates and starts machines
- Runs health checks

## 5. Handle Common Issues

### If image is too big (>8GB):
```bash
# Scale memory during deployment
fly scale memory 2048  # if 512MB default fails
fly deploy --remote-only
```

### If OOM during startup:
```bash
# Upgrade to performance CPU for more memory
fly scale vm performance-1x --memory 4096
fly deploy --remote-only
```

### If health checks fail:
```bash
# Check logs for errors
fly logs

# SSH into machine to debug
fly ssh console
# Inside: curl localhost:7860/health
```

## 6. Verify Deployment

```bash
# Check status
fly status

# Test health endpoint
curl https://your-app-name.fly.dev/health
# Should return: {"status":"healthy"}

# View logs
fly logs

# Check machine details
fly machines list
```

## 7. Configure Twilio

In Twilio Console:
1. Go to Phone Numbers → Manage → Active numbers
2. Click your phone number
3. Set Webhook URL: `https://your-app-name.fly.dev/call`
4. Set HTTP method: `POST`
5. Save configuration

## 8. Test End-to-End

```bash
# Call your Twilio number
# Should hear: "Thank you for calling Thérapie Clinic, how can I help you today?"

# Monitor call in logs
fly logs --follow
```

## 9. Cost Management

```bash
# Scale to single machine (cheaper)
fly scale count 1

# Check current pricing
fly status
# shared-cpu-1x:2048MB ≈ $4.50/month
# performance-1x:4096MB ≈ $12/month

# Auto-stop saves money (configured in fly.toml)
# Machines stop after ~5 minutes of inactivity
```

## 10. Future Updates

```bash
# Deploy code changes
fly deploy --remote-only

# Scale resources if needed (edit fly.toml, then deploy)
fly scale memory 4096  # or
fly scale vm performance-1x --memory 4096

# View secrets
fly secrets list

# Add new secrets
fly secrets set NEW_API_KEY=value
```

## Common Gotchas We Encountered

1. Image size: Started at 9GB, fixed with CPU-only PyTorch
2. Memory: Needed 2GB+ for sentence-transformers
3. Qdrant auth: 403 errors with wrong API key
4. Health checks: Needed longer grace period for startup
5. Auto-stop: Normal behavior, saves money

Your app is now deployed and will auto-scale based on call volume! 🚀
