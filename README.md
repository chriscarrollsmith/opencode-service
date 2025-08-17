## OpenCode API Service (Modal)

Secure, key-protected FastAPI service on Modal that runs `opencode` jobs against per-session workspaces stored on a Modal Volume.

OpenCode is a powerful AI coding agent that supports 75+ LLM providers including Anthropic Claude, OpenAI GPT, Google Gemini, and local models. This service wraps OpenCode in a secure API that can be deployed on Modal for scalable cloud execution.

### What you get

- **Secured API**: All endpoints require `X-API-Key` except `GET /health`.
- **Ephemeral sessions**: Upload files, run a job, fetch outputs, then delete.
- **Background execution**: Jobs run in a worker function; state tracked in Modal Dicts.
- **Scheduled cleanup**: Expired sessions are removed periodically.

### Prerequisites

- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) installed
- Project dependencies installed: `uv sync`
- Modal CLI authenticated (`uv run modal setup`).
- Generated API key and Anthropic API key stored as a Modal secret:
  ```bash
  export API_KEY=$(openssl rand -hex 32)
  export ANTHROPIC_API_KEY='your-anthropic-api-key'
  uv run modal secret create opencode-secret API_KEY=$API_KEY ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY
  echo "API_KEY=$API_KEY" >> .env
  echo "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY" >> .env
  ```
  
  Note: OpenCode supports multiple AI providers. If you want to use a different provider, you'll need to set the appropriate API keys (e.g., `OPENAI_API_KEY` for OpenAI) and specify the model in your API requests.

### Setup and deployment

1) Deploy the service:

```bash
uv run modal deploy main.py
```

2) (Optional) Run the built-in smoke test. It will use the deployed URL and your `API_KEY`:

```bash
uv run modal run main.py::main  # `uv` loads the .env file
```

### Authentication

- Send `X-API-Key: $API_KEY` with every request (except `GET /health`).
- The expected key is read from the Modal secret `opencode-secret` (env var `API_KEY`).

### Endpoints

- `GET /health` (public): returns `{ status: "healthy" }`.
- `POST /session` → create session
  - Body: `{ files: [{ path, content }], ttl_seconds? }`
  - Returns: `{ session_id, created_at, expires_at, input_files: [...] }`
- `POST /job?session_id=...` → queue a job
  - Body: `{ prompt, model?, timeout_s? }`
  - Model format: `provider/model` (e.g., `anthropic/claude-3-5-sonnet-latest`, `openai/gpt-4`)
  - Returns: `{ job_id, session_id, status, created_at }`
- `GET /job/status?job_id=...` → job status/details
- `GET /session/results?session_id=...` → list outputs from latest successful job
- `GET /download?session_id=...&file_path=...` → stream a single output file
- `DELETE /session?session_id=...` → delete a session and its files

### cURL examples

Assuming `BASE` is your deployed URL (e.g. `https://<org>--opencode-service-fastapi-app.modal.run`):

1. Create a session

```bash
# Create a session
curl -sS -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -X POST "$BASE/session" \
  -d '{
    "files": [
      {"path": "README.md", "content": "# My Project\n"},
      {"path": "src/math_utils.py", "content": "def add(a,b):\n    return a+b\n"}
    ],
    "ttl_seconds": 3600
  }'
```

2. Run a job

```bash
# Run a job (default model: anthropic/claude-3-5-sonnet-latest)
curl -sS -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -X POST "$BASE/job?session_id=<SESSION_ID>" \
  -d '{"prompt": "Add a subtract function to src/math_utils.py"}'

# Run a job with specific model
curl -sS -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -X POST "$BASE/job?session_id=<SESSION_ID>" \
  -d '{"prompt": "Add a subtract function to src/math_utils.py", "model": "openai/gpt-4"}'
```

3. Poll status until the job is finished

```bash
# Poll status
curl -sS -H "X-API-Key: $API_KEY" \
  "$BASE/job/status?job_id=<JOB_ID>"
```

4. List results

```bash
# List results
curl -sS -H "X-API-Key: $API_KEY" \
  "$BASE/session/results?session_id=<SESSION_ID>"
```

5. Download a file

```bash
# Download a file
curl -sS -H "X-API-Key: $API_KEY" \
  -o out.md \
  "$BASE/download?session_id=<SESSION_ID>&file_path=README.md"
```

6. Delete the session

```bash
# Delete session
curl -sS -H "X-API-Key: $API_KEY" \
  -X DELETE "$BASE/session?session_id=<SESSION_ID>"
```

### Data & limits

- Files are stored under a Modal Volume at `/data/sessions/<session_id>/`.
- Default limits are defined in `main.py` (e.g., max files, max session size, job timeout).
- Session TTL is extended on job submission; expired sessions are removed by the scheduled cleanup.

### Secrets

- `opencode-secret`: must contain `API_KEY` (used by the FastAPI auth middleware) and provider API keys based on which models you plan to use:
  - `ANTHROPIC_API_KEY` for Anthropic Claude models
  - `OPENAI_API_KEY` for OpenAI GPT models  
  - `GEMINI_API_KEY` for Google Gemini models
  - `OPENROUTER_API_KEY` for OpenRouter models
  - And other provider-specific keys as needed

### Supported Models

OpenCode supports models from 75+ providers. Common examples:
- Anthropic: `anthropic/claude-4-sonnet`, `anthropic/claude-4-opus`
- OpenAI: `openai/gpt-5`
- Google: `google/gemini-2.5-pro`
- OpenRouter: `openrouter/moonshotai/kimi-k2`
- And many more including AWS Bedrock, Azure OpenAI, Groq, local models, etc.

### Notes

- Ensure your local `API_KEY` matches the deployed secret value when using the smoke test or cURL.