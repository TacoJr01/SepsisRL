# Module 3: Deploying Environments

## Three Things a Space Gives You

Every HF Space running an OpenEnv environment provides three access methods:

| Component | What it provides | How to access |
|-----------|------------------|---------------|
| **Server** | Running environment endpoint | `https://<username>-<space-name>.hf.space` |
| **Repository** | Pip-installable Python package | `pip install git+https://huggingface.co/spaces/<space>` |
| **Registry** | Docker container image | `docker pull registry.hf.space/<space>:latest` |

One deployment. Three ways to use it.

## Local Development with Uvicorn

The fastest iteration loop: clone a Space and run it locally.

```bash
# Clone from HF Space
git clone https://huggingface.co/spaces/openenv/echo-env
cd echo-env

# Install and run
uv sync
uv run server
```

Or with uvicorn directly:

```bash
uvicorn echo_env.server.app:app --host 0.0.0.0 --port 8000 --reload
```

The `--reload` flag restarts the server when you change code. Essential for development.

Test it:
```bash
curl http://localhost:8000/health
# {"status": "healthy"}
```

Connect from Python:
```python
with EchoEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset()
```

## Docker Deployment

Docker gives you isolation and reproducibility.

### Pull from a Space's registry:
```bash
docker pull registry.hf.space/openenv-echo-env:latest
docker run -d -p 8000:8000 registry.hf.space/openenv-echo-env:latest
```

### Build from source:
```bash
git clone https://huggingface.co/spaces/openenv/echo-env
cd echo-env
docker build -t my-echo-env:latest -f server/Dockerfile .
docker run -d -p 8000:8000 my-echo-env:latest
```

### With environment variables:
```bash
docker run -d -p 8000:8000 \
    -e WORKERS=4 \
    -e MAX_CONCURRENT_ENVS=100 \
    my-echo-env:latest
```

## Deploying to HF Spaces

### Using `openenv push`

The fastest path from local code to a running endpoint:

```bash
cd my_env
openenv push --repo-id username/my-env
```

Your environment is now live:
- **API endpoint:** `https://username-my-env.hf.space`
- **Web UI:** `https://username-my-env.hf.space/web`
- **API docs:** `https://username-my-env.hf.space/docs`
- **Health check:** `https://username-my-env.hf.space/health`

### The `openenv.yaml` Manifest

Controls Space settings:

```yaml
name: my_env
version: "1.0.0"
description: My custom environment
```

### Environment Variables

Configure via Space Settings → Variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `WORKERS` | 4 | Uvicorn worker processes |
| `PORT` | 8000 | Server port |
| `HOST` | 0.0.0.0 | Bind address |
| `MAX_CONCURRENT_ENVS` | 100 | Max WebSocket sessions |

### Hardware Options

| Tier | vCPU | RAM | Cost |
|------|------|-----|------|
| CPU Basic (Free) | 2 | 16GB | Free |
| CPU Upgrade | 8 | 32GB | $0.03/hr |

Free tier handles ~128 concurrent sessions — enough for development and demos.

## The Full Workflow

```
1. openenv init my_env       # Scaffold
2. Edit server/environment.py # Implement logic
3. uv run server              # Test locally
4. openenv push               # Deploy to HF Spaces
5. pip install git+https://huggingface.co/spaces/username/my-env  # Install client
```

## Choosing Your Access Method

| Method | Use when | Pros | Cons |
|--------|----------|------|------|
| **Remote Space** | Quick testing, low volume | Zero setup | Network latency |
| **Local Docker** | Development, high throughput | Full control, no network | Requires Docker |
| **Local Uvicorn** | Fast iteration | Fastest reload | No isolation |

## What's Next

In the [notebook](notebook.ipynb), you'll clone the Echo environment, modify it, run it locally, and deploy your modified version to HF Spaces.

**Key takeaway:** One Space gives you a running server, a pip-installable package, and a Docker image. `openenv push` gets you there in one command.
