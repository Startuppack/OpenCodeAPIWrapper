import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

load_dotenv()

app = FastAPI(title="OpenCodeToAPI", version="1.0.0")

OVH_API_KEY = os.environ.get("OVH_AI_KEY") or os.environ.get("OVH_API_KEY")
if not OVH_API_KEY:
    raise RuntimeError("OVH_AI_KEY (ou OVH_API_KEY) est requis — définissez-le dans l'environnement ou dans un fichier .env")
OVH_BASE_URL = os.environ.get(
    "OVH_BASE_URL", "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"
)
OVH_MODEL = os.environ.get("OVH_MODEL", "Qwen2.5-Coder-32B-Instruct")


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

    # Fichiers appartenant à l'utilisateur — corrigé après useradd
    subprocess.run(
        ["chown", "-R", f"{home_dir.name}:{home_dir.name}", str(config_dir)],
        check=True,
        capture_output=True,
    )


def _run_opencode(username: str, work_dir: Path, prompt: str) -> str:
    env = {
        **os.environ,
        "HOME": str(work_dir),
        "USER": username,
        "OPENAI_API_KEY": OVH_API_KEY,
        "OPENAI_BASE_URL": OVH_BASE_URL,
    }

    result = subprocess.run(
        ["su", "-s", "/bin/bash", username, "-c",
         f"opencode run --dangerously-skip-permissions {_shell_quote(prompt)}"],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
        cwd=str(work_dir),
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"opencode exited with code {result.returncode}:\n{result.stderr}"
        )

    return result.stdout, result.stderr


def _shell_quote(s: str) -> str:
    """Échappe une chaîne pour un shell bash."""
    return "'" + s.replace("'", "'\\''") + "'"


@app.post("/process")
async def process(
    prompt: str = Form(..., description="Prompt à envoyer à OpenCode"),
    html_file: UploadFile = File(..., description="Fichier HTML lourd à traiter"),
):
    """
    Reçoit un fichier HTML et un prompt, crée un utilisateur temporaire,
    exécute OpenCode avec la clé OVH AI, renvoie le résultat puis supprime
    l'utilisateur et tous ses fichiers.
    """
    transaction_id = uuid.uuid4().hex[:12]
    username = f"oc_{transaction_id}"

    try:
        # 1. Créer l'utilisateur système
        _create_system_user(username)
        home_dir = Path(f"/home/{username}")

        # 2. Écrire le fichier HTML dans le répertoire de travail
        html_content = await html_file.read()
        html_path = home_dir / "index.html"
        html_path.write_bytes(html_content)
        subprocess.run(
            ["chown", f"{username}:{username}", str(html_path)],
            check=True,
            capture_output=True,
        )

        # 3. Configurer OpenCode pour cet utilisateur
        _write_opencode_config(home_dir)

        # 4. Exécuter OpenCode
        output, stderr_output = await asyncio.get_event_loop().run_in_executor(
            None, _run_opencode, username, home_dir, prompt
        )

        # 5. Lire le fichier HTML modifié par OpenCode (modifié en place)
        modified_html: str | None = None
        if html_path.exists():
            try:
                modified_html = html_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                pass

        return JSONResponse(
            content={
                "transaction_id": transaction_id,
                "result": output,
                "stderr": stderr_output,
                "html": modified_html,
            }
        )

    except subprocess.CalledProcessError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur système : {exc.stderr.decode(errors='replace')}",
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        # 5. Toujours supprimer l'utilisateur et ses fichiers
        _delete_system_user(username)


@app.get("/health")
async def health():
    return {"status": "ok"}
