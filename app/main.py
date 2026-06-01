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
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("opencode-wrapper")

OVH_API_KEY = os.environ.get("OVH_AI_KEY") or os.environ.get("OVH_API_KEY")
if not OVH_API_KEY:
    raise RuntimeError("OVH_AI_KEY (or OVH_API_KEY) is required — set it in the environment or in a .env file")
OVH_BASE_URL = os.environ.get(
    "OVH_BASE_URL", "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"
)
OVH_MODEL = os.environ.get("OVH_MODEL", "Qwen3-Coder-30B-A3B-Instruct")
DEBUG_KEEP_USER_DATA = os.environ.get("DEBUG_KEEP_USER_DATA", "false").lower() == "true"
REPO_DIFF_MAX_CHARS = int(os.environ.get("REPO_DIFF_MAX_CHARS", "60000"))


async def _validate_model() -> None:
    """Check that OVH_MODEL is available on the configured endpoint."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{OVH_BASE_URL.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {OVH_API_KEY}"},
            )
            resp.raise_for_status()
            available = [m["id"] for m in resp.json().get("data", [])]
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"Failed to list OVH models ({e.response.status_code}): {e.response.text}") from e
    except Exception as e:
        raise RuntimeError(f"Unable to reach OVH endpoint to validate model: {e}") from e

    if OVH_MODEL not in available:
        raise RuntimeError(
            f"Model '{OVH_MODEL}' not found on OVH endpoint.\n"
            f"Available models: {', '.join(available)}"
        )
    log.info("Model validated: %s", OVH_MODEL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _validate_model()
    yield


app = FastAPI(title="OpenCodeToAPI", version="1.0.0", lifespan=lifespan)


class RepoProcessRequest(BaseModel):
    repo_url: str = Field(..., description="SSH Git URL, for example git@github.com:owner/repo.git")
    ssh_private_key: str = Field(..., description="Private SSH key allowed to read/write the repository")
    instruction: str = Field(..., description="Instruction to execute in the cloned repository")
    branch: str | None = Field(None, description="Branch to clone. Defaults to the repository default branch")
    push: bool = Field(False, description="Commit and push changes back to the branch")
    commit_message: str | None = Field(None, description="Commit message to use when push=true")
    git_user_name: str = Field("OpenCode API Wrapper", description="Git author name for commits")
    git_user_email: str = Field("opencode-api-wrapper@example.local", description="Git author email for commits")
    max_diff_chars: int | None = Field(None, ge=1000, le=500000, description="Maximum diff characters returned")


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
                    "timeout": 600000,
                },
                "models": {
                    OVH_MODEL: {
                        "name": OVH_MODEL,
                        "tool_call": True,
                        "limit": {"context": 262144, "output": 65536},
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


def _validate_git_ssh_url(repo_url: str) -> None:
    if not repo_url.strip():
        raise ValueError("repo_url is required")

    allowed = (
        repo_url.startswith("git@")
        or repo_url.startswith("ssh://")
    )
    if not allowed:
        raise ValueError("repo_url must be an SSH Git URL such as git@github.com:owner/repo.git")


def _validate_git_ref(ref: str | None, field_name: str) -> None:
    if ref is None:
        return
    if not re.fullmatch(r"[A-Za-z0-9._/\-]+", ref):
        raise ValueError(f"{field_name} contains unsupported characters")
    if ref.startswith("/") or ".." in ref or ref.endswith(".lock"):
        raise ValueError(f"{field_name} is not a safe git ref")


def _setup_ssh_key(home_dir: Path, username: str, private_key: str) -> Path:
    ssh_dir = home_dir / ".ssh"
    ssh_dir.mkdir(mode=0o700, exist_ok=True)

    key_file = ssh_dir / "id_repo"
    key_text = private_key.strip() + "\n"
    key_file.write_text(key_text)
    key_file.chmod(0o600)

    known_hosts = ssh_dir / "known_hosts"
    known_hosts.touch(mode=0o600, exist_ok=True)

    subprocess.run(
        ["chown", "-R", f"{username}:{username}", str(ssh_dir)],
        check=True,
        capture_output=True,
    )
    return key_file


def _repo_env(home_dir: Path, key_file: Path) -> dict[str, str]:
    known_hosts = home_dir / ".ssh" / "known_hosts"
    return {
        **os.environ,
        "HOME": str(home_dir),
        "GIT_SSH_COMMAND": (
            f"ssh -i {_shell_quote(str(key_file))} "
            "-o IdentitiesOnly=yes "
            "-o StrictHostKeyChecking=accept-new "
            f"-o UserKnownHostsFile={_shell_quote(str(known_hosts))}"
        ),
    }


def _run_as_user(
    username: str,
    command: str,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout: int = 600,
) -> subprocess.CompletedProcess[str]:
    exports = []
    if env:
        for key in ("HOME", "GIT_SSH_COMMAND"):
            if key in env:
                exports.append(f"export {key}={_shell_quote(env[key])};")
    shell_command = " ".join([*exports, command])
    return subprocess.run(
        ["su", "-s", "/bin/bash", username, "-c", shell_command],
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def _run_checked_as_user(
    username: str,
    command: str,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout: int = 600,
) -> subprocess.CompletedProcess[str]:
    result = _run_as_user(username, command, cwd, env=env, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {command}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def _build_repo_prompt(user_instruction: str) -> str:
    return (
        "You are operating inside a cloned Git repository. "
        "Inspect the project files, then implement this instruction: "
        f"{user_instruction.rstrip()}. "
        "Use the available edit tools to modify files directly in the repository. "
        "Run lightweight checks when the project exposes obvious commands. "
        "Do not ask questions or wait for clarification. "
        "If there are multiple reasonable interpretations, make the most useful concrete change."
    )


def _run_opencode_repo(username: str, home_dir: Path, repo_dir: Path, prompt: str) -> tuple[str, str]:
    env = {
        **os.environ,
        "HOME": str(home_dir),
        "USER": username,
        "OPENAI_API_KEY": OVH_API_KEY,
        "OPENAI_BASE_URL": OVH_BASE_URL,
    }

    full_prompt = _build_repo_prompt(prompt)
    log.debug("[opencode-repo] Running with prompt: %s", full_prompt)

    import threading
    timeout_seconds = int(os.environ.get("OPENCODE_TIMEOUT", "600"))

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    def _stream(pipe, lines: list[str], level: str) -> None:
        for raw in pipe:
            line = raw.rstrip("\n")
            lines.append(line)
            if level == "stderr":
                log.debug("[opencode-repo|stderr] %s", line)
            else:
                log.info("[opencode-repo] %s", line)
        pipe.close()

    proc = subprocess.Popen(
        ["su", "-s", "/bin/bash", username, "-c",
         f"opencode run --dangerously-skip-permissions {_shell_quote(full_prompt)}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=str(repo_dir),
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
        log.error("[opencode-repo] Timed out after %ds", timeout_seconds)
        raise RuntimeError(f"opencode timed out after {timeout_seconds}s")

    t_out.join()
    t_err.join()

    stdout_text = "\n".join(stdout_lines)
    stderr_text = "\n".join(stderr_lines)

    if proc.returncode != 0:
        log.error("[opencode-repo] Exited with code %d", proc.returncode)
        raise RuntimeError(f"opencode exited with code {proc.returncode}:\n{stderr_text}")

    return stdout_text, stderr_text


# Replace any attribute value containing a base64 data URI (src=, poster=, href=, data=, etc.)
_ATTR_B64_PATTERN = re.compile(
    r'((?:src|poster|href|data|srcset|background)=")([^"]*data:[^;"\s]+;base64,[A-Za-z0-9+/=\n][^"]*)(")' ,
    re.IGNORECASE,
)
# Replace base64 data URIs inside url() (e.g., background-image:url(data:image/...;base64,...))
# Handles multiline and various whitespace
_URL_B64_PATTERN = re.compile(
    r'url\(\s*data:[^;]+;base64,[A-Za-z0-9+/=\n\s]+\s*\)',
    re.IGNORECASE | re.MULTILINE,
)
# Bare data URIs not in attributes (in style content, etc.)
_BARE_B64_PATTERN = re.compile(
    r'data:[^;]+;base64,[A-Za-z0-9+/=\n\s]+(?=[\'"\s);,])',
    re.IGNORECASE | re.MULTILINE,
)
# External scripts only (<script src="..."></script>) — inline scripts may contain SSR/hydration data
_EXTERNAL_SCRIPT_PATTERN = re.compile(r'<script\b[^>]*\bsrc\s*=[^>]*>\s*</script>', re.IGNORECASE)
# Inline SVG: keep <svg ...> and </svg> tags, replace only the inner content
_SVG_INNER_PATTERN = re.compile(r'(<svg\b[^>]*>)([\s\S]*?)(</svg>)', re.IGNORECASE)


def _minify_css(css: str) -> str:
    """Minify CSS: remove comments, extra whitespace, but keep it readable."""
    # Remove /* */ comments
    css = re.sub(r'/\*[\s\S]*?\*/', '', css)
    # Remove leading/trailing whitespace from rules
    css = re.sub(r'{\s+', '{', css)
    css = re.sub(r'\s+}', '}', css)
    # Remove whitespace around selectors and properties
    css = re.sub(r';\s+', ';', css)
    css = re.sub(r':\s+', ':', css)
    css = re.sub(r',\s+', ',', css)
    # Remove trailing semicolon before }
    css = re.sub(r';\s*}', '}', css)
    return css.strip()


def _remove_empty_attrs(html: str) -> str:
    """Remove empty id and class attributes."""
    # Remove id="" or id=''
    html = re.sub(r'\s+id=(["\'])(["\'])', '', html, flags=re.IGNORECASE)
    # Remove class="" or class=''
    html = re.sub(r'\s+class=(["\'])(["\'])', '', html, flags=re.IGNORECASE)
    return html


def _preprocess_html(html: str) -> tuple[str, dict[str, str]]:
    """Lighten the HTML before sending it to OpenCode:
    - strip base64 attribute values (src, poster, href, srcset…) and replace with a placeholder
    - replace the inner content of inline <svg> blocks with a placeholder (tags are preserved)
    - replace external <script src="..."> tags with placeholders
    - <style> blocks are kept intact so the AI can read and modify CSS
    """
    placeholders: dict[str, str] = {}
    counter = 0

    def _replace(match: re.Match) -> str:
        nonlocal counter
        key = f"__PLACEHOLDER_{counter}__"
        placeholders[key] = match.group(0)
        counter += 1
        return key

    def _replace_attr_b64(match: re.Match) -> str:
        """Keep the attribute name and quotes, replace only the value."""
        nonlocal counter
        key = f"__PLACEHOLDER_{counter}__"
        placeholders[key] = match.group(2)  # stocke la valeur brute (sans guillemets)
        counter += 1
        return f"{match.group(1)}{key}{match.group(3)}"

    def _replace_svg_inner(match: re.Match) -> str:
        """Keep <svg ...> and </svg>, replace only the inner content."""
        nonlocal counter
        inner = match.group(2)
        if not inner.strip():
            return match.group(0)  # Empty SVG, nothing to do
        key = f"__PLACEHOLDER_{counter}__"
        placeholders[key] = inner
        counter += 1
        return f"{match.group(1)}{key}{match.group(3)}"

    def _replace_url_b64(match: re.Match) -> str:
        """Replace base64 data URIs inside url()."""
        nonlocal counter
        key = f"__PLACEHOLDER_{counter}__"
        placeholders[key] = match.group(0)
        counter += 1
        return f"url(__PLACEHOLDER_{counter - 1}__)"

    def _replace_bare_b64(match: re.Match) -> str:
        """Replace bare base64 data URIs."""
        nonlocal counter
        key = f"__PLACEHOLDER_{counter}__"
        placeholders[key] = match.group(0)
        counter += 1
        return f"__PLACEHOLDER_{counter - 1}__"

    html = _ATTR_B64_PATTERN.sub(_replace_attr_b64, html)
    html = _URL_B64_PATTERN.sub(_replace_url_b64, html)
    html = _BARE_B64_PATTERN.sub(_replace_bare_b64, html)
    html = _SVG_INNER_PATTERN.sub(_replace_svg_inner, html)
    html = _EXTERNAL_SCRIPT_PATTERN.sub(_replace, html)

    # Minify CSS inside <style> blocks
    def _minify_style_block(match: re.Match) -> str:
        tag_open = match.group(1)
        css = match.group(2)
        tag_close = match.group(3)
        return tag_open + _minify_css(css) + tag_close

    html = re.sub(r'(<style[^>]*>)([\s\S]*?)(</style>)', _minify_style_block, html, flags=re.IGNORECASE)

    # Remove empty id and class attributes
    html = _remove_empty_attrs(html)

    # Inject an explanatory comment at the very top so the AI understands the placeholders
    banner = (
        "<!-- PREPROCESSING NOTE: This file has been optimized for AI processing. "
        "Tokens matching __PLACEHOLDER_N__ are stand-ins for large binary data "
        "(base64-encoded images, inline SVG bodies, external <script> tags). "
        "<style> blocks are fully intact and can be read and modified normally. "
        "They are NOT errors or missing content — the HTML structure is complete and correct. "
        "Treat every __PLACEHOLDER_N__ token as opaque content: do NOT remove, modify, or comment "
        "it out unless the element that directly contains it is itself being removed. "
        "Focus only on the visible HTML structure, text and CSS to fulfil the requested task. -->"
    )
    html = banner + "\n" + html
    return html, placeholders


_BANNER_PATTERN = re.compile(r'<!-- PREPROCESSING NOTE:.*?-->\n?', re.DOTALL)


def _postprocess_html(html: str, placeholders: dict[str, str]) -> str:
    # Remove the explanatory banner injected for the AI
    html = _BANNER_PATTERN.sub("", html)
    for key, value in placeholders.items():
        if key in html:
            html = html.replace(key, value)
        else:
            log.warning("Placeholder %s not found in HTML after OpenCode run — skipped (may have been intentionally removed)", key)
    return html


def _build_prompt(user_prompt: str) -> str:
    return (
        f"Read index.html, then apply the following modifications: {user_prompt.rstrip()}. "
        "IMPORTANT: The file contains __PLACEHOLDER_N__ tokens (e.g. __PLACEHOLDER_0__, __PLACEHOLDER_1__…). "
        "These are intentional stand-ins for large binary data (base64 images, SVG bodies, CSS). "
        "The HTML structure is real and complete — do NOT treat the file as corrupted or incomplete. "
        "Leave every __PLACEHOLDER_N__ token untouched unless you are explicitly removing the element that contains it. "
        "Use the edit tool to make targeted, minimal changes only to the relevant parts. "
        "Do not rewrite the entire file. "
        "NEVER ask questions, NEVER say you cannot do something, NEVER ask for clarification. "
        "If you cannot find the exact element, make your best guess based on the context and apply the change anyway using inline styles or by adding a <style> block. "
        "Always produce a concrete edit to the file, no matter what."
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


def _pretty_print_html(html: str) -> str:
    """Break a minified single-line HTML into multiple lines so OpenCode can read it.
    Uses html.parser-based indentation when possible, falls back to tag-boundary splitting."""
    try:
        from html.parser import HTMLParser

        class _Breaker(HTMLParser):
            def __init__(self) -> None:
                super().__init__(convert_charrefs=False)
                self.parts: list[str] = []
                self._last_end = 0
                self._src = ""

            def feed_src(self, src: str) -> str:
                self._src = src
                self.feed(src)
                # Append any trailing text after the last tag
                self.parts.append(src[self._last_end:])
                return "\n".join(p for p in self.parts if p)

            def handle_starttag(self, tag: str, attrs: list) -> None:
                self._flush_text()

            def handle_endtag(self, tag: str) -> None:
                self._flush_text()

            def handle_startendtag(self, tag: str, attrs: list) -> None:
                self._flush_text()

            def _flush_text(self) -> None:
                pos = self.getpos()
                # getpos returns (line, col) — col is 0-based in the source
                # We rebuild from raw source using offset tracking
                pass

        # Simple but effective: insert a newline before every tag opening
        import re as _re
        broken = _re.sub(r'>\s*<', '>\n<', html)
        return broken
    except Exception:
        return html


@app.post("/process")
async def process(
    prompt: str = Form(..., description="Prompt to send to OpenCode"),
    html_file: UploadFile = File(..., description="HTML file to process"),
):
    transaction_id = uuid.uuid4().hex[:12]
    username = f"oc_{transaction_id}"
    log.info("[%s] New request — prompt: %r", transaction_id, prompt)

    try:
        _create_system_user(username)
        home_dir = Path(f"/home/{username}")

        html_content = await html_file.read()
        log.debug("[%s] Original HTML: %d bytes", transaction_id, len(html_content))

        html_str = html_content.decode("utf-8", errors="replace")
        processed_html, placeholders = _preprocess_html(html_str)
        processed_html = _pretty_print_html(processed_html)
        log.debug("[%s] After preprocessing: %d bytes, %d placeholders", transaction_id, len(processed_html), len(placeholders))

        html_path = home_dir / "index.html"

        processed_html_bytes = processed_html.encode("utf-8")
        preprocessed_md5 = _md5(processed_html_bytes)
        html_path.write_bytes(processed_html_bytes)
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
                changed = modified_md5 != preprocessed_md5
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
            detail=f"System error: {exc.stderr.decode(errors='replace')}",
        ) from exc
    except RuntimeError as exc:
        log.error("[%s] RuntimeError: %s", transaction_id, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if DEBUG_KEEP_USER_DATA:
            log.warning("[%s] DEBUG_KEEP_USER_DATA=true — user '%s' and files kept at /home/%s", transaction_id, username, username)
        else:
            _delete_system_user(username)


@app.post("/process-repo")
async def process_repo(request: RepoProcessRequest):
    transaction_id = uuid.uuid4().hex[:12]
    username = f"oc_{transaction_id}"
    log.info("[%s] New repo request — repo: %s", transaction_id, request.repo_url)

    try:
        _validate_git_ssh_url(request.repo_url)
        _validate_git_ref(request.branch, "branch")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        _create_system_user(username)
        home_dir = Path(f"/home/{username}")
        repo_dir = home_dir / "repo"

        key_file = _setup_ssh_key(home_dir, username, request.ssh_private_key)
        git_env = _repo_env(home_dir, key_file)

        clone_parts = ["git", "clone"]
        if request.branch:
            clone_parts.extend(["--branch", request.branch])
        clone_parts.extend([request.repo_url, str(repo_dir)])
        clone_command = " ".join(_shell_quote(part) for part in clone_parts)

        clone_result = _run_checked_as_user(
            username,
            clone_command,
            home_dir,
            env=git_env,
            timeout=int(os.environ.get("GIT_CLONE_TIMEOUT", "600")),
        )
        log.info("[%s] Repository cloned", transaction_id)

        _write_opencode_config(home_dir)

        output, stderr_output = await asyncio.get_event_loop().run_in_executor(
            None, _run_opencode_repo, username, home_dir, repo_dir, request.instruction
        )

        status_result = _run_checked_as_user(
            username,
            "git status --porcelain",
            repo_dir,
            env=git_env,
        )
        changed = bool(status_result.stdout.strip())

        # Include untracked files in the returned diff without staging real content.
        _run_as_user(username, "git add -N .", repo_dir, env=git_env)
        diff_stat = _run_checked_as_user(
            username,
            "git diff --stat",
            repo_dir,
            env=git_env,
        ).stdout
        diff = _run_checked_as_user(
            username,
            "git diff -- .",
            repo_dir,
            env=git_env,
            timeout=120,
        ).stdout

        max_diff_chars = request.max_diff_chars or REPO_DIFF_MAX_CHARS
        diff_truncated = len(diff) > max_diff_chars
        if diff_truncated:
            diff = diff[:max_diff_chars] + "\n\n[diff truncated]"

        git_result: dict[str, Any] = {
            "pushed": False,
            "commit_sha": None,
            "branch": None,
            "push_stdout": "",
            "push_stderr": "",
        }

        if request.push and changed:
            _run_checked_as_user(username, f"git config user.name {_shell_quote(request.git_user_name)}", repo_dir, env=git_env)
            _run_checked_as_user(username, f"git config user.email {_shell_quote(request.git_user_email)}", repo_dir, env=git_env)
            _run_checked_as_user(username, "git add -A", repo_dir, env=git_env)

            commit_message = request.commit_message or f"Apply OpenCode instruction {transaction_id}"
            _run_checked_as_user(
                username,
                f"git commit -m {_shell_quote(commit_message)}",
                repo_dir,
                env=git_env,
                timeout=120,
            )

            current_branch = _run_checked_as_user(
                username,
                "git rev-parse --abbrev-ref HEAD",
                repo_dir,
                env=git_env,
            ).stdout.strip()
            if current_branch == "HEAD":
                raise RuntimeError("Cannot push from detached HEAD. Provide a branch in the request.")

            push_timeout = int(os.environ.get("GIT_PUSH_TIMEOUT", "600"))
            push_cmd = f"git push origin HEAD:{_shell_quote(current_branch)}"
            try:
                push_result = _run_checked_as_user(
                    username, push_cmd, repo_dir, env=git_env, timeout=push_timeout,
                )
            except subprocess.CalledProcessError as push_exc:
                # Le dépôt <slug>/site.git survit aux teardowns : son `main` distant
                # peut avoir divergé du clone → push rejeté (non fast-forward,
                # « fetch first »). Le site étant regénéré intégralement, notre arbre
                # fait foi : on intègre le distant avec la stratégie `ours` (le distant
                # devient un ancêtre, notre contenu est conservé tel quel) puis on
                # repousse. On ne fait ce repli que sur un vrai rejet de push.
                stderr_txt = (push_exc.stderr or b"")
                if isinstance(stderr_txt, bytes):
                    stderr_txt = stderr_txt.decode(errors="replace")
                if "rejected" not in stderr_txt and "fetch first" not in stderr_txt and "non-fast-forward" not in stderr_txt:
                    raise
                log.warning("[%s] push rejeté (divergence distante) — réconciliation -s ours puis retry", transaction_id)
                _run_checked_as_user(username, f"git fetch origin {_shell_quote(current_branch)}", repo_dir, env=git_env, timeout=120)
                _run_checked_as_user(
                    username,
                    f"git merge -s ours --no-edit FETCH_HEAD -m {_shell_quote('Reconcile diverged remote (generated site wins)')}",
                    repo_dir, env=git_env, timeout=120,
                )
                push_result = _run_checked_as_user(
                    username, push_cmd, repo_dir, env=git_env, timeout=push_timeout,
                )
            commit_sha = _run_checked_as_user(
                username,
                "git rev-parse HEAD",
                repo_dir,
                env=git_env,
            ).stdout.strip()
            git_result = {
                "pushed": True,
                "commit_sha": commit_sha,
                "branch": current_branch,
                "push_stdout": push_result.stdout,
                "push_stderr": push_result.stderr,
            }

        log.info("[%s] Repo request done — changed=%s pushed=%s", transaction_id, changed, git_result["pushed"])

        return JSONResponse(
            content={
                "transaction_id": transaction_id,
                "changed": changed,
                "status": status_result.stdout,
                "diff_stat": diff_stat,
                "diff": diff,
                "diff_truncated": diff_truncated,
                "result": output,
                "stderr": stderr_output,
                "git": git_result,
                "clone_stdout": clone_result.stdout,
                "clone_stderr": clone_result.stderr,
            }
        )

    except subprocess.CalledProcessError as exc:
        log.error("[%s] CalledProcessError: %s", transaction_id, exc.stderr)
        raise HTTPException(
            status_code=500,
            detail=f"System error: {exc.stderr.decode(errors='replace')}",
        ) from exc
    except subprocess.TimeoutExpired as exc:
        log.error("[%s] Timeout: %s", transaction_id, exc)
        raise HTTPException(status_code=500, detail=f"Command timed out: {exc}") from exc
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
