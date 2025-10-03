"""
OpenCode API client for managing sessions, jobs, and file operations.

This module provides a reusable client for interacting with the OpenCode Service API,
handling session lifecycle, job execution, polling, result retrieval, and downloads.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional, List, Dict, Any
from datetime import datetime

import requests
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# --- Pydantic Models mirroring server contracts (main.py) ---

class FileInput(BaseModel):
    path: str = Field(..., description="Relative path of the file in the workspace.")
    content: str = Field(..., description="UTF-8 encoded content of the file.")


class SessionCreateRequest(BaseModel):
    files: List[FileInput]
    ttl_seconds: int = Field(3600, gt=0)


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
    model: Optional[str] = Field("openai/gpt-5", description="Provider-prefixed model, e.g., 'openai/gpt-5'")
    timeout_s: int = Field(600, gt=0, le=1800)


class OutputFileResult(BaseModel):
    path: str
    size_bytes: int
    sha256: str
    download_url: Optional[str] = None


class JobRunResponse(BaseModel):
    job_id: str
    session_id: str
    status: str
    created_at: datetime


class JobStatusResponse(BaseModel):
    job_id: str
    session_id: str
    status: str  # "queued", "running", "succeeded", "failed", "timed_out"
    error: Optional[str] = None
    prompt: str
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    output_files: Optional[List[OutputFileResult]] = None
    stdout: Optional[str] = None


class SessionResultsResponse(BaseModel):
    session_id: str
    latest_job_id: Optional[str] = None
    output_files: List[OutputFileResult]
    result: Optional[str] = None


class OpenCodeClient:
    """
    Client for interacting with the OpenCode Service API.

    Handles session management, job execution, polling, and cleanup.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        cleanup_on_exit: bool = False,
        default_model: Optional[str] = None,
    ) -> None:
        """
        Initialize the OpenCode Service API client.

        Args:
            api_key: API key for authentication. Defaults to OPENCODE_API_KEY env var.
            base_url: Base URL for the API. Defaults to OPENCODE_BASE_URL env var.
            cleanup_on_exit: If True, delete the session on context exit.
            default_model: Preferred model provider string, e.g., "openai/gpt-5".
        """
        api_key_value = api_key or os.getenv("OPENCODE_API_KEY") or os.getenv("API_KEY")
        base_url_value = base_url or os.getenv("OPENCODE_BASE_URL")

        if not api_key_value or not base_url_value:
            raise ValueError("OPENCODE_API_KEY (or API_KEY) and OPENCODE_BASE_URL must be set")

        self.api_key: str = api_key_value
        self.base_url: str = base_url_value.rstrip("/")
        self.cleanup_on_exit = cleanup_on_exit
        self.session_id: Optional[str] = None
        self.default_model: Optional[str] = default_model

        self.headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }

    # --- Session ---
    def create_session(self, files: List[FileInput], ttl_seconds: int = 3600) -> str:
        logger.info("Creating session on OpenCode Service API...")

        req = SessionCreateRequest(files=files, ttl_seconds=ttl_seconds)
        resp = requests.post(f"{self.base_url}/session", headers=self.headers, json=req.model_dump())
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to create session: {resp.status_code} - {resp.text}")

        parsed = SessionCreateResponse.model_validate(resp.json())
        self.session_id = parsed.session_id
        logger.info("Created session: %s", self.session_id)
        return self.session_id

    # --- Jobs ---
    def run_job(self, prompt: str, timeout_s: int = 600, model: Optional[str] = None) -> str:
        if not self.session_id:
            raise RuntimeError("No active session. Call create_session() first.")

        req = JobRunRequest(prompt=prompt, timeout_s=timeout_s, model=model or self.default_model or "openai/gpt-5")
        resp = requests.post(
            f"{self.base_url}/job",
            headers=self.headers,
            params={"session_id": self.session_id},
            json=req.model_dump(),
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to start job: {resp.status_code} - {resp.text}")

        parsed = JobRunResponse.model_validate(resp.json())
        logger.info("Started job: %s", parsed.job_id)
        return parsed.job_id

    def wait_for_job_completion(self, job_id: str, timeout_s: int = 600, poll_interval_s: float = 5.0) -> JobStatusResponse:
        start = time.time()
        while True:
            resp = requests.get(f"{self.base_url}/job/status", headers=self.headers, params={"job_id": job_id})
            if resp.status_code != 200:
                raise RuntimeError(f"Failed to get job status: {resp.status_code} - {resp.text}")
            status = JobStatusResponse.model_validate(resp.json())
            if status.status == "succeeded":
                logger.info("Job %s completed successfully", job_id)
                return status
            if status.status == "failed":
                raise RuntimeError(f"Job failed: {status.error or 'Unknown error'}")
            if status.status == "timed_out":
                raise RuntimeError("Job timed out on server")

            elapsed = time.time() - start
            if elapsed > timeout_s:
                raise RuntimeError(f"Job timed out after {elapsed:.1f}s")
            time.sleep(poll_interval_s)

    # --- Results ---
    def get_session_results(self) -> SessionResultsResponse:
        if not self.session_id:
            raise RuntimeError("No active session")
        resp = requests.get(
            f"{self.base_url}/session/results",
            headers=self.headers,
            params={"session_id": self.session_id},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to get results: {resp.status_code} - {resp.text}")
        return SessionResultsResponse.model_validate(resp.json())

    # --- Download ---
    def download_file(self, file_path: str, timeout_s: int = 60) -> bytes:
        if not self.session_id:
            raise RuntimeError("No active session")
        resp = requests.get(
            f"{self.base_url}/download",
            headers=self.headers,
            params={"session_id": self.session_id, "file_path": file_path},
            timeout=timeout_s,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to download file: {resp.status_code} - {resp.text}")
        return resp.content

    # --- Delete ---
    def delete_session(self) -> None:
        if not self.session_id or not self.cleanup_on_exit:
            return
        try:
            resp = requests.delete(
                f"{self.base_url}/session",
                headers=self.headers,
                params={"session_id": self.session_id},
                timeout=30,
            )
            if resp.status_code in (200, 204):
                logger.info("Deleted session: %s", self.session_id)
            elif resp.status_code == 404:
                logger.info("Session %s already deleted or expired", self.session_id)
            else:
                logger.warning("Failed to delete session: %s - %s", resp.status_code, resp.text)
        finally:
            self.session_id = None

    # --- High-level convenience ---
    def execute_job(
        self,
        prompt: str,
        inputs: Optional[Dict[str, Any]] = None,
        input_files: Optional[List[FileInput]] = None,
        output_dir: Optional[str] = "out",
        timeout_s: int = 600,
        model: Optional[str] = None,
    ) -> SessionResultsResponse:
        files = input_files or []
        try:
            self.create_session(files, ttl_seconds=max(timeout_s * 2, 3600))
            final_prompt = prompt.format(**inputs) if inputs else prompt
            job_id = self.run_job(final_prompt, timeout_s=timeout_s, model=model)
            self.wait_for_job_completion(job_id, timeout_s=timeout_s)
            results = self.get_session_results()

            if output_dir and results.output_files:
                for f in results.output_files:
                    content = self.download_file(f.path)
                    target_path = os.path.join(output_dir, f.path)
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    # Detect if looks like text
                    try:
                        text = content.decode("utf-8")
                        mode = "w"
                        data_to_write = text
                        with open(target_path, mode, encoding="utf-8") as out_f:
                            out_f.write(data_to_write)
                    except UnicodeDecodeError:
                        mode_b = "wb"
                        with open(target_path, mode_b) as out_fb:
                            out_fb.write(content)

            return results
        finally:
            self.delete_session()

    def __enter__(self) -> "OpenCodeClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.delete_session()



if __name__ == "__main__":
    # Example usage with asyncio semaphore for concurrency control
    import asyncio
    import dotenv

    dotenv.load_dotenv()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Allow up to 3 concurrent jobs
    semaphore = asyncio.Semaphore(3)

    prompt_template = "Read {filename} and write a short story to {output_filename}"

    async def run_one(i: int):
        async with semaphore:
            def sync_job() -> str:
                client = OpenCodeClient(
                    api_key=os.getenv("OPENCODE_API_KEY"),
                    base_url=os.getenv("OPENCODE_BASE_URL"),
                    cleanup_on_exit=True,
                    default_model=os.getenv("OPENCODE_DEFAULT_MODEL") or "openai/gpt-5",
                )
                result = client.execute_job(
                    prompt=prompt_template,
                    inputs={"filename": f"topic_{i}.md", "output_filename": f"story_{i}.md"},
                    input_files=[FileInput(
                        path=f"topic_{i}.md",
                        content="The story should be about a cat."
                    )],
                    output_dir=f"output_{i}",
                    timeout_s=180,
                )
                return result.result or ""

            # Run the blocking client job in a background thread
            return await asyncio.to_thread(sync_job)

    async def run_tasks():
        tasks = [run_one(i) for i in range(5)]
        return await asyncio.gather(*tasks, return_exceptions=True)

    results = asyncio.run(run_tasks())
    for idx, r in enumerate(results):
        if isinstance(r, Exception):
            print(f"Job {idx} failed: {r}")
        else:
            print(f"Job {idx} completed. Result snippet: {r[:120]}")
