# main.py

import hashlib
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import modal
from fastapi import FastAPI, HTTPException, Response, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- Constants and Configuration ---

# Service-level configuration
APP_NAME = "claude-code-service"
VOLUME_NAME = "claude-code-data"
SESSIONS_DICT_NAME = "claude-code-sessions"
JOBS_DICT_NAME = "claude-code-jobs"
SECRET_NAME = "claude-code-secret"

# Operational limits and defaults
MAX_SESSION_SIZE_MB = 25
MAX_FILE_COUNT = 200
DEFAULT_SESSION_TTL_S = 3600  # 1 hour
DEFAULT_JOB_TIMEOUT_S = 600   # 10 minutes
MAX_JOB_TIMEOUT_S = 1800      # 30 minutes

# Volume paths
SESSIONS_ROOT = Path("/data/sessions")

# --- Modal App Setup ---

app = modal.App(APP_NAME)

# Volume for persistent file storage (input/output)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Dictionaries for state management (metadata)
sessions_dict = modal.Dict.from_name(SESSIONS_DICT_NAME, create_if_missing=True)
jobs_dict = modal.Dict.from_name(JOBS_DICT_NAME, create_if_missing=True)

# Image definition
image = (
    modal.Image.debian_slim(python_version="3.13")
    .run_commands(
        "apt-get update",
        "apt-get install -y curl git",
        "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -",
        "apt-get install -y nodejs",
        "npm install -g @anthropic-ai/claude-code",
    )
    .pip_install("fastapi", "pydantic") # fastapi for web_asgi
)

# --- Pydantic Models for API ---

class FileInput(BaseModel):
    path: str = Field(..., description="Relative path of the file in the workspace.")
    content: str = Field(..., description="UTF-8 encoded content of the file.")

class SessionCreateRequest(BaseModel):
    files: List[FileInput]
    ttl_seconds: int = Field(DEFAULT_SESSION_TTL_S, gt=0)

class FileInfo(BaseModel):
    path: str
    size_bytes: int

class SessionCreateResponse(BaseModel):
    session_id: str
    created_at: datetime
    expires_at: datetime
    input_files: List[FileInfo]

class JobRunRequest(BaseModel):
    prompt: str
    model: Optional[str] = Field(None, description="Claude model to use (e.g., 'claude-4-sonnet')")
    timeout_s: int = Field(DEFAULT_JOB_TIMEOUT_S, gt=0, le=MAX_JOB_TIMEOUT_S)

class JobRunResponse(BaseModel):
    job_id: str
    session_id: str
    status: str = "queued"
    created_at: datetime

class OutputFileResult(BaseModel):
    path: str
    size_bytes: int
    sha256: str
    download_url: Optional[str] = None

class JobStatusResponse(BaseModel):
    job_id: str
    session_id: str
    status: str # "queued", "running", "succeeded", "failed", "timed_out"
    error: Optional[str] = None
    prompt: str
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    output_files: Optional[List[OutputFileResult]] = None

class SessionResultsResponse(BaseModel):
    session_id: str
    latest_job_id: Optional[str] = None
    output_files: List[OutputFileResult]

# --- Core Worker Function ---

@app.function(
    image=image,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    timeout=MAX_JOB_TIMEOUT_S + 60,  # Allow buffer over job timeout
    max_containers=10, # Adjust as needed
)
def execute_job(session_id: str, job_id: str, request: Dict[str, Any]):
    """The background worker that executes a claude-code run."""
    session_input_dir = SESSIONS_ROOT / session_id / "input"
    session_output_dir = SESSIONS_ROOT / session_id / "output"
    now = datetime.now(timezone.utc)

    # 1. Update job status to "running"
    job_data = {
        **jobs_dict[job_id],
        "status": "running",
        "started_at": now.isoformat(),
    }
    jobs_dict[job_id] = job_data

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)

            # 2. Materialize workspace by copying from Volume
            shutil.copytree(session_input_dir, workspace, dirs_exist_ok=True)

            # 3. Execute Claude Code (with simple retry for transient failures)
            prompt = request["prompt"]
            model = request.get("model")
            timeout_s = request.get("timeout_s", DEFAULT_JOB_TIMEOUT_S)

            cmd = ["claude", "-p", prompt, "--output-format", "json", "--debug"]
            if model:
                cmd.extend(["-m", model])

            env = os.environ.copy()

            last_err: Optional[str] = None
            max_attempts = 2
            for attempt in range(1, max_attempts + 1):
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(workspace),
                    env=env,
                    timeout=timeout_s
                )
                if result.returncode == 0:
                    break
                last_err = result.stderr
                if attempt < max_attempts:
                    time.sleep(1.0)
            if result.returncode != 0:
                raise RuntimeError(f"Claude Code failed after {max_attempts} attempts:\n{last_err}")

            # 4. Process results
            if result.returncode != 0:
                raise RuntimeError(f"Claude Code failed:\n{result.stderr}")

            # 5. Clear previous output and copy new files to Volume
            if os.path.exists(session_output_dir):
                shutil.rmtree(session_output_dir)
            shutil.copytree(workspace, session_output_dir, dirs_exist_ok=True)
            
            # 6. Generate manifest of output files
            output_files_manifest = []
            for f_path in Path(session_output_dir).glob("**/*"):
                if f_path.is_file():
                    content = f_path.read_bytes()
                    output_files_manifest.append(
                        {
                            "path": str(f_path.relative_to(session_output_dir)),
                            "size_bytes": len(content),
                            "sha256": hashlib.sha256(content).hexdigest(),
                        }
                    )

            # 7. Update job status to "succeeded"
            finished_at = datetime.now(timezone.utc)
            job_data.update({
                "status": "succeeded",
                "finished_at": finished_at.isoformat(),
                "output_files": output_files_manifest
            })
            jobs_dict[job_id] = job_data
            
            # Update session with the latest job ID
            session_data = sessions_dict.get(session_id)
            if session_data:
                session_data["latest_job_id"] = job_id
                sessions_dict[session_id] = session_data

            # Persist output files to the shared volume so other functions can read them
            volume.commit()

    except subprocess.TimeoutExpired:
        job_data.update({
            "status": "timed_out",
            "error": f"Job exceeded the timeout of {request.get('timeout_s', DEFAULT_JOB_TIMEOUT_S)} seconds.",
            "finished_at": datetime.now(timezone.utc).isoformat(),
        })
        jobs_dict[job_id] = job_data
    except Exception as e:
        job_data.update({
            "status": "failed",
            "error": str(e),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        })
        jobs_dict[job_id] = job_data

# --- Secured FastAPI ASGI App ---

@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=MAX_JOB_TIMEOUT_S + 60,
    secrets=[modal.Secret.from_name(SECRET_NAME, required_keys=["API_KEY"])],
)
@modal.asgi_app()
def fastapi_app():
    """Serve the service as a secured FastAPI app with API key authentication."""

    secure_app = FastAPI(
        title="Claude Code Service (Secured)",
        description="Secure API for Claude Code service endpoints",
        version="1.0.0",
    )

    secure_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @secure_app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        if request.url.path in ["/health", "/openapi.json"]:
            return await call_next(request)
        api_key = request.headers.get("X-API-Key")
        expected_key = os.environ.get("API_KEY")
        if expected_key and (not api_key or api_key != expected_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        return await call_next(request)

    @secure_app.get("/health")
    async def health():
        return {"status": "healthy"}

    # --- API Endpoints (secured) ---

    @secure_app.post("/session", response_model=SessionCreateResponse)
    async def create_session(req: SessionCreateRequest):
        session_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=req.ttl_seconds)
        session_dir = SESSIONS_ROOT / session_id

        if len(req.files) > MAX_FILE_COUNT:
            raise HTTPException(status_code=400, detail=f"Exceeded max file count of {MAX_FILE_COUNT}")
        total_size = sum(len(f.content.encode()) for f in req.files)
        if total_size > MAX_SESSION_SIZE_MB * 1024 * 1024:
            raise HTTPException(status_code=400, detail=f"Exceeded max session size of {MAX_SESSION_SIZE_MB}MB")

        input_dir = session_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "output").mkdir(exist_ok=True)

        input_files_info = []
        for file_input in req.files:
            rel = Path(file_input.path)
            if rel.is_absolute() or ".." in rel.parts:
                raise HTTPException(status_code=400, detail=f"Invalid file path: {file_input.path}")
            file_path = input_dir / rel
            file_path.parent.mkdir(parents=True, exist_ok=True)
            content_bytes = file_input.content.encode('utf-8')
            file_path.write_bytes(content_bytes)
            input_files_info.append(FileInfo(path=file_input.path, size_bytes=len(content_bytes)))

        volume.commit()

        sessions_dict[session_id] = {
            "created_at": now.isoformat(),
            "expires_at": expires_at.isoformat(),
            "size_bytes": total_size,
        }

        return SessionCreateResponse(
            session_id=session_id,
            created_at=now,
            expires_at=expires_at,
            input_files=input_files_info,
        )

    @secure_app.post("/job", response_model=JobRunResponse)
    async def run_job(session_id: str, request: JobRunRequest):
        if session_id not in sessions_dict:
            raise HTTPException(status_code=404, detail="Session not found")

        job_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc)

        jobs_dict[job_id] = {
            "job_id": job_id,
            "session_id": session_id,
            "status": "queued",
            "prompt": request.prompt,
            "created_at": now.isoformat(),
        }

        session_data = sessions_dict.get(session_id, {})
        new_expires = now + timedelta(seconds=DEFAULT_SESSION_TTL_S)
        current_expires = datetime.fromisoformat(session_data.get("expires_at")) if session_data.get("expires_at") else new_expires
        session_data["expires_at"] = max(current_expires, new_expires).isoformat()
        sessions_dict[session_id] = session_data

        execute_job.spawn(session_id, job_id, request.model_dump())

        return JobRunResponse(job_id=job_id, session_id=session_id, created_at=now)

    @secure_app.get("/job/status", response_model=JobStatusResponse)
    async def get_job_status(job_id: str):
        job_data = jobs_dict.get(job_id)
        if not job_data:
            raise HTTPException(status_code=404, detail="Job not found")
        return JobStatusResponse(**job_data)

    @secure_app.delete("/session")
    async def delete_session(session_id: str):
        if session_id not in sessions_dict:
            raise HTTPException(status_code=404, detail="Session not found")

        session_dir = SESSIONS_ROOT / session_id
        try:
            if session_dir.exists():
                shutil.rmtree(session_dir)
            volume.commit()
        except FileNotFoundError:
            pass

        del sessions_dict[session_id]
        return Response(status_code=204)

    @secure_app.get("/session/results", response_model=SessionResultsResponse)
    async def list_session_results(session_id: str):
        session_data = sessions_dict.get(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")

        latest_job_id = session_data.get("latest_job_id")
        if not latest_job_id:
            return SessionResultsResponse(session_id=session_id, output_files=[])

        job_data = jobs_dict.get(latest_job_id)
        if not job_data or job_data.get("status") != "succeeded":
            return SessionResultsResponse(session_id=session_id, latest_job_id=latest_job_id, output_files=[])

        files = []
        for f in job_data.get("output_files", []):
            # Placeholder for direct download URL; client can construct using /download
            files.append(
                OutputFileResult(
                    **{**f, "download_url": None}
                )
            )

        return SessionResultsResponse(
            session_id=session_id,
            latest_job_id=latest_job_id,
            output_files=files,
        )

    @secure_app.get("/download")
    async def download_file(session_id: str, file_path: str):
        session_data = sessions_dict.get(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")

        # Ensure we see the latest committed files from other containers
        try:
            volume.reload()
        except Exception:
            pass

        base_dir = (SESSIONS_ROOT / session_id / "output")
        full_path = (base_dir / file_path)

        # Path traversal protection
        try:
            base_dir_resolved = base_dir.resolve()
            full_path_resolved = full_path.resolve()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid file path")
        if not str(full_path_resolved).startswith(str(base_dir_resolved)):
            raise HTTPException(status_code=400, detail="Invalid file path")

        if not full_path_resolved.exists() or not full_path_resolved.is_file():
            raise HTTPException(status_code=404, detail="File not found in session output")

        def file_iterator(chunk_size: int = 1024 * 64):
            with open(full_path_resolved, "rb") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk

        headers = {"Content-Disposition": f"attachment; filename={full_path_resolved.name}"}
        return StreamingResponse(file_iterator(), media_type="application/octet-stream", headers=headers)

    return secure_app

# --- Scheduled Cleanup Function ---

@app.function(
    schedule=modal.Cron("*/10 * * * *"), # Runs every 10 minutes
    image=image,
    volumes={"/data": volume},
)
def cleanup_expired_sessions():
    """Periodically cleans up expired sessions from the volume and dict."""
    print("Running scheduled cleanup...")
    now = datetime.now(timezone.utc)
    expired_sessions = []
    
    # This iteration is safe in Python 3.
    for session_id, data in sessions_dict.items():
        expires_at = datetime.fromisoformat(data["expires_at"])
        if now > expires_at:
            expired_sessions.append(session_id)
    
    print(f"Found {len(expired_sessions)} expired sessions to delete.")
    for session_id in expired_sessions:
        print(f"Deleting session {session_id}...")
        try:
            # Delete files from volume
            session_path = SESSIONS_ROOT / session_id
            if session_path.exists():
                shutil.rmtree(session_path)
            # Delete metadata
            del sessions_dict[session_id]
        except Exception as e:
            print(f"Error deleting session {session_id}: {e}")
    
    if expired_sessions:
        volume.commit()
    print("Cleanup complete.")


# --- Local Entrypoint for Testing ---

@app.local_entrypoint()
def main():
    import requests
    from dotenv import load_dotenv
    
    load_dotenv()
    
    """A local test client for the Claude Code service."""
    print("🚀 Testing Claude Code Service...")

    # 1. Create a session
    print("\n[1] Creating session...")
    test_files = [
        FileInput(
            path="src/math_utils.py",
            content="def add(a, b):\n    return a + b\n"
        ),
        FileInput(
            path="README.md",
            content="# My Project\n"
        )
    ]
    create_req = SessionCreateRequest(files=test_files)
    # Use get_web_url() to avoid deprecation
    base = fastapi_app.get_web_url()
    headers = {"X-API-Key": os.environ.get("API_KEY", "")}

    create_response = requests.post(f"{base}/session", json=create_req.model_dump(), headers=headers)
    if create_response.status_code != 200:
        print(f"❌ Error creating session: {create_response.status_code} {create_response.text}")
        create_response.raise_for_status()
    session_create = SessionCreateResponse.model_validate(create_response.json())
    session_id = session_create.session_id
    print(f"✅ Session created: {session_id}")

    # 2. Run a job
    print("\n[2] Running job...")
    run_req = JobRunRequest(prompt="Add a function to subtract two numbers in math_utils.py")
    run_response = requests.post(f"{base}/job", params={"session_id": session_id}, json=run_req.model_dump(), headers=headers)
    if run_response.status_code != 200:
        print(f"❌ Error running job: {run_response.status_code} {run_response.text}")
        run_response.raise_for_status()
    run_job_parsed = JobRunResponse.model_validate(run_response.json())
    job_id = run_job_parsed.job_id
    print(f"✅ Job '{job_id}' queued.")

    # 3. Poll for status
    print("\n[3] Polling job status...")
    while True:
        status_response = requests.get(f"{base}/job/status", params={"job_id": job_id}, headers=headers)
        if status_response.status_code != 200:
            print(f"❌ Error polling job status: {status_response.status_code} {status_response.text}")
            status_response.raise_for_status()
        status_parsed = JobStatusResponse.model_validate(status_response.json())
        status = status_parsed.status
        print(f"   ...status is '{status}'")
        if status in ["succeeded", "failed", "timed_out"]:
            print(f"✅ Job finished with status: {status}")
            if status == 'failed':
                print(f"   Error: {status_parsed.error}")
            break
        time.sleep(3)

    # 4. List results if successful
    if status == 'succeeded':
        print("\n[4] Listing results...")
        results_response = requests.get(f"{base}/session/results", params={"session_id": session_id}, headers=headers)
        if results_response.status_code != 200:
            print(f"❌ Error listing results: {results_response.status_code} {results_response.text}")
            results_response.raise_for_status()
        results_parsed = SessionResultsResponse.model_validate(results_response.json())
        print("   Output files:")
        for f in results_parsed.output_files:
            print(f"   - {f.path} ({f.size_bytes} bytes)")


    # 5. Download a file
    print("\n[5] Downloading file...")
    download_response = requests.get(f"{base}/download", params={"session_id": session_id, "file_path": "README.md"}, headers=headers)
    if download_response.status_code != 200:
        print(f"❌ Error downloading file: {download_response.status_code} {download_response.text}")
        download_response.raise_for_status()
    print(f"✅ File downloaded: {download_response.content}")

    # 6. Clean up
    print("\n[6] Deleting session...")
    delete_response = requests.delete(f"{base}/session", params={"session_id": session_id}, headers=headers)
    if delete_response.status_code != 204:
        print(f"❌ Error deleting session: {delete_response.status_code} {delete_response.text}")
        delete_response.raise_for_status()
    print(f"✅ Session '{session_id}' deleted.")
    
    print("\n🚀 Test complete.")