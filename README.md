## Claude Code API Service (Modal)

Secure, key-protected FastAPI service on Modal that runs `@anthropic-ai/claude-code` jobs against per-session workspaces stored on a Modal Volume.

### What you get

- **Secured API**: All endpoints require `X-API-Key` except `GET /health`.
- **Ephemeral sessions**: Upload files, run a job, fetch outputs, then delete.
- **Background execution**: Jobs run in a worker function; state tracked in Modal Dicts.
- **Scheduled cleanup**: Expired sessions are removed periodically.

### Prerequisites

- Modal CLI installed and authenticated (`modal token set ...`).
- Create an API key and store it as a Modal secret.

### Setup and deployment

1) Generate an API key and store it as a Modal secret along with your Anthropic API key (keep the key safe):

```bash
export API_KEY=$(openssl rand -hex 32)
export ANTHROPIC_API_KEY='your-anthropic-api-key'
modal secret create claude-code-secret API_KEY=$API_KEY ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY
```

2) Deploy the service:

```bash
modal deploy main.py
```

3) (Optional) Run the built-in smoke test. It will use the deployed URL and your `API_KEY`:

```bash
API_KEY=$API_KEY modal run main.py::main
```

The deployment exposes a single ASGI app. In code, its URL is available as `fastapi_app.web_url`.

### Authentication

- Send `X-API-Key: $API_KEY` with every request (except `GET /health`).
- The expected key is read from the Modal secret `claude-code-auth-secret` (env var `API_KEY`).

### Endpoints

- `GET /health` (public): returns `{ status: "healthy" }`.
- `POST /session` → create session
  - Body: `{ files: [{ path, content }], ttl_seconds? }`
  - Returns: `{ session_id, created_at, expires_at, input_files: [...] }`
- `POST /job?session_id=...` → queue a job
  - Body: `{ prompt, model?, timeout_s? }`
  - Returns: `{ job_id, session_id, status, created_at }`
- `GET /job/status?job_id=...` → job status/details
- `GET /session/results?session_id=...` → list outputs from latest successful job
- `GET /download?session_id=...&file_path=...` → stream a single output file
- `DELETE /session?session_id=...` → delete a session and its files

### cURL examples

Assuming `BASE` is your deployed URL (e.g. `https://<org>--claude-code-service-fastapi-app.modal.run`):

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
# Run a job
curl -sS -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -X POST "$BASE/job?session_id=<SESSION_ID>" \
  -d '{"prompt": "Add a subtract function to src/math_utils.py"}'
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

- `claude-code-secret`: must contain `API_KEY` (used by the FastAPI auth middleware) and `ANTHROPIC_API_KEY` (used by the worker).

### Notes

- Ensure your local `API_KEY` matches the deployed secret value when using the smoke test or cURL.