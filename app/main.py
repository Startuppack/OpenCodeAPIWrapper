import asyncio
import hashlib
import json
import logging
import os
import re
import subprocess
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("opencode-wrapper")

OVH_API_KEY = os.environ.get("OVH_AI_KEY") or os.environ.get("OVH_API_KEY")
if not OVH_API_KEY:
    raise RuntimeError("OVH_AI_KEY (ou OVH_API_KEY) est requis — définissez-le dans l'environnement ou dans un fichier .env")
OVH_BASE_URL = os.environ.get(
    "OVH_BASE_URL", "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"
)
OVH_MODEL = os.environ.get("OVH_MODEL", "Qwen3-Coder-30B-A3B-Instruct")
DEBUG_KEEP_USER_DATA = os.environ.get("DEBUG_KEEP_USER_DATA", "false").lower() == "true"


async def _validate_model() -> None:
    """Vérifie que OVH_MODEL est disponible sur l'endpoint configuré."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{OVH_BASE_URL.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {OVH_API_KEY}"},
            )
            resp.raise_for_status()
            available = [m["id"] for m in resp.json().get("data", [])]
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"Impossible de lister les modèles OVH ({e.response.status_code}): {e.response.text}") from e
    except Exception as e:
        raise RuntimeError(f"Impossible de contacter l'endpoint OVH pour valider le modèle: {e}") from e

    if OVH_MODEL not in available:
        raise RuntimeError(
            f"Modèle '{OVH_MODEL}' introuvable sur l'endpoint OVH.\n"
            f"Modèles disponibles : {', '.join(available)}"
        )
    log.info("Model validated: %s", OVH_MODEL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _validate_model()
    yield


app = FastAPI(title="OpenCodeToAPI", version="1.0.0", lifespan=lifespan)


def _create_system_user(username: str) -> None:
    subprocess.run(
        ["useradd", "-m", "-s", "/bin/bash", username],
        check=True,
        capture_output=True,
    )


def _delete_system_user(username: str) -> None:
    subprocess.run(
        ["userdel", "-r", "-f", username],
        capture_output=True,
    )


def _write_opencode_config(home_dir: Path) -> None:
    config_dir = home_dir / ".config" / "opencode"
    config_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "ovh": {
                "api": "openai-compatible",
                "name": "OVH AI",
                "options": {
                    "apiKey": OVH_API_KEY,
                    "baseURL": OVH_BASE_URL,
                },
                "models": {
                    OVH_MODEL: {
                        "name": OVH_MODEL,
                        "tool_call": True,
                        "limit": {"context": 131072, "output": 32768},
                    }
                },
            }
        },
        "model": f"ovh/{OVH_MODEL}",
        "autoupdate": False,
        "share": "disabled",
        "snapshot": False,
    }

    config_file = config_dir / "opencode.json"
    config_file.write_text(json.dumps(config, indent=2))

    subprocess.run(
        ["chown", "-R", f"{home_dir.name}:{home_dir.name}", str(config_dir)],
        check=True,
        capture_output=True,
    )


_B64_PATTERN = re.compile(r'data:[^;"\s]+;base64,[A-Za-z0-9+/=]+')
_SCRIPT_PATTERN = re.compile(r'<script\b[^>]*>[\s\S]*?</script>', re.IGNORECASE)
_STYLE_PATTERN = re.compile(r'<style\b[^>]*>[\s\S]*?</style>', re.IGNORECASE)


def _preprocess_html(html: str) -> tuple[str, dict[str, str]]:
    """Strip base64 data URIs, script blocks and style blocks. Returns processed HTML + placeholders."""
    placeholders: dict[str, str] = {}
    counter = 0

    def _replace(match: re.Match) -> str:
        nonlocal counter
        key = f"__PLACEHOLDER_{counter}__"
        placeholders[key] = match.group(0)
        counter += 1
        return key

    html = _B64_PATTERN.sub(_replace, html)
    html = _SCRIPT_PATTERN.sub(_replace, html)
    html = _STYLE_PATTERN.sub(_replace, html)
    return html, placeholders


def _postprocess_html(html: str, placeholders: dict[str, str]) -> str:
    for key, value in placeholders.items():
        html = html.replace(key, value)
    return html


def _build_prompt(user_prompt: str) -> str:
    return (
        f"Read index.html, then apply the following modifications: {user_prompt.rstrip()}. "
        "Use the edit tool to make targeted, minimal changes only to the relevant parts. "
        "Do not rewrite the entire file. Do not ask questions."
    )


def _run_opencode(username: str, work_dir: Path, prompt: str) -> tuple[str, str]:
    env = {
        **os.environ,
        "HOME": str(work_dir),
        "USER": username,
        "OPENAI_API_KEY": OVH_API_KEY,
        "OPENAI_BASE_URL": OVH_BASE_URL,
    }

    full_prompt = _build_prompt(prompt)
    log.debug("[opencode] Running with prompt: %s", full_prompt)

    import threading
    timeout_seconds = int(os.environ.get("OPENCODE_TIMEOUT", "600"))

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    def _stream(pipe, lines: list[str], level: str) -> None:
        for raw in pipe:
            line = raw.rstrip("\n")
            lines.append(line)
            if level == "stderr":
                log.debug("[opencode|stderr] %s", line)
            else:
                log.info("[opencode] %s", line)
        pipe.close()

    proc = subprocess.Popen(
        ["su", "-s", "/bin/bash", username, "-c",
         f"opencode run --dangerously-skip-permissions {_shell_quote(full_prompt)}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=str(work_dir),
    )

    t_out = threading.Thread(target=_stream, args=(proc.stdout, stdout_lines, "stdout"), daemon=True)
    t_err = threading.Thread(target=_stream, args=(proc.stderr, stderr_lines, "stderr"), daemon=True)
    t_out.start()
    t_err.start()

    try:
        proc.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        proc.kill()
        t_out.join(timeout=2)
        t_err.join(timeout=2)
        log.error("[opencode] Timed out after %ds", timeout_seconds)
        raise RuntimeError(f"opencode timed out after {timeout_seconds}s — the file may be too large or the model too slow")

    t_out.join()
    t_err.join()

    stdout_text = "\n".join(stdout_lines)
    stderr_text = "\n".join(stderr_lines)

    if proc.returncode != 0:
        log.error("[opencode] Exited with code %d", proc.returncode)
        raise RuntimeError(f"opencode exited with code {proc.returncode}:\n{stderr_text}")

    return stdout_text, stderr_text


def _shell_quote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def _md5(content: bytes) -> str:
    return hashlib.md5(content).hexdigest()


@app.post("/process")
async def process(
    prompt: str = Form(..., description="Prompt à envoyer à OpenCode"),
    html_file: UploadFile = File(..., description="Fichier HTML à traiter"),
):
    transaction_id = uuid.uuid4().hex[:12]
    username = f"oc_{transaction_id}"
    log.info("[%s] New request — prompt: %r", transaction_id, prompt)

    try:
        _create_system_user(username)
        home_dir = Path(f"/home/{username}")

        html_content = await html_file.read()
        original_md5 = _md5(html_content)
        log.debug("[%s] Original HTML: %d bytes, md5=%s", transaction_id, len(html_content), original_md5)

        html_str = html_content.decode("utf-8", errors="replace")
        processed_html, placeholders = _preprocess_html(html_str)
        log.debug("[%s] After preprocessing: %d bytes, %d placeholders", transaction_id, len(processed_html), len(placeholders))

        html_path = home_dir / "index.html"

        html_path.write_text(processed_html, encoding="utf-8")
        subprocess.run(
            ["chown", f"{username}:{username}", str(html_path)],
            check=True,
            capture_output=True,
        )

        _write_opencode_config(home_dir)

        output, stderr_output = await asyncio.get_event_loop().run_in_executor(
            None, _run_opencode, username, home_dir, prompt
        )

        modified_html: str | None = None
        changed = False

        if html_path.exists():
            try:
                modified_bytes = html_path.read_bytes()
                modified_md5 = _md5(modified_bytes)
                changed = modified_md5 != original_md5
                modified_html = _postprocess_html(modified_bytes.decode("utf-8", errors="replace"), placeholders)
                log.info("[%s] index.html after run: %d bytes, md5=%s, changed=%s", transaction_id, len(modified_bytes), modified_md5, changed)
                if not changed:
                    log.warning("[%s] index.html is identical to input — OpenCode made no changes", transaction_id)
                    log.warning("[%s] OpenCode stdout: %s", transaction_id, output)
            except Exception as e:
                log.error("[%s] Error reading index.html: %s", transaction_id, e)
        else:
            log.error("[%s] index.html not found after OpenCode execution", transaction_id)
            log.error("[%s] OpenCode stdout: %s", transaction_id, output)
            log.error("[%s] OpenCode stderr: %s", transaction_id, stderr_output)

        log.info("[%s] Done — changed=%s", transaction_id, changed)

        return JSONResponse(
            content={
                "transaction_id": transaction_id,
                "changed": changed,
                "result": output,
                "stderr": stderr_output,
                "html": modified_html,
            }
        )

    except subprocess.CalledProcessError as exc:
        log.error("[%s] CalledProcessError: %s", transaction_id, exc.stderr)
        raise HTTPException(
            status_code=500,
            detail=f"Erreur système : {exc.stderr.decode(errors='replace')}",
        ) from exc
    except RuntimeError as exc:
        log.error("[%s] RuntimeError: %s", transaction_id, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if DEBUG_KEEP_USER_DATA:
            log.warning("[%s] DEBUG_KEEP_USER_DATA=true — user '%s' and files kept at /home/%s", transaction_id, username, username)
        else:
            _delete_system_user(username)


@app.get("/health")
async def health():
    return {"status": "ok"}
