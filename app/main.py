import asyncio
import hashlib
import json
import logging
import os
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
OVH_MODEL = os.environ.get("OVH_MODEL", "Qwen2.5-Coder-32B-Instruct")


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
                        "limit": {"context": 32768, "output": 4096},
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


def _build_prompt(user_prompt: str) -> str:
    """Wraps the user prompt to ensure OpenCode always works on index.html."""
    return (
        f"{user_prompt.rstrip()}. "
        "Apply all changes directly to the file index.html and save it."
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

    result = subprocess.run(
        ["su", "-s", "/bin/bash", username, "-c",
         f"opencode run --dangerously-skip-permissions {_shell_quote(full_prompt)}"],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
        cwd=str(work_dir),
    )

    log.debug("[opencode] stdout:\n%s", result.stdout)
    if result.stderr:
        log.debug("[opencode] stderr:\n%s", result.stderr)

    if result.returncode != 0:
        log.error("[opencode] Exited with code %d:\n%s", result.returncode, result.stderr)
        raise RuntimeError(
            f"opencode exited with code {result.returncode}:\n{result.stderr}"
        )

    return result.stdout, result.stderr


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

        html_path = home_dir / "index.html"
        html_path.write_bytes(html_content)
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
                modified_html = modified_bytes.decode("utf-8", errors="replace")

                if changed:
                    log.info("[%s] HTML was modified (md5 %s → %s)", transaction_id, original_md5, modified_md5)
                else:
                    log.warning("[%s] HTML was NOT modified — OpenCode made no changes (md5 unchanged: %s)", transaction_id, original_md5)
                    log.warning("[%s] OpenCode stdout: %s", transaction_id, output)
            except Exception as e:
                log.error("[%s] Error reading modified HTML: %s", transaction_id, e)
        else:
            log.error("[%s] index.html not found after OpenCode execution", transaction_id)

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
        _delete_system_user(username)


@app.get("/health")
async def health():
    return {"status": "ok"}
