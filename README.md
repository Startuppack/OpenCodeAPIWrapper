# OpenCodeAPIWrapper

REST API that exposes [OpenCode](https://github.com/anomalyco/opencode) as an HTTP service. Send an HTML file and a prompt — OpenCode modifies the file and the API returns the modified HTML.

## Requirements

- Docker + Docker Compose
- An OVH AI Endpoints API key

## Setup

```bash
cp .env.example .env
# Fill in OVH_AI_KEY in .env
```

## Start

```bash
docker compose up --build -d
```

## Endpoints

### `GET /health`
Check the service is running.

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

### `POST /process`
Send an HTML file and a prompt. OpenCode modifies the file and returns the resulting HTML.

**Parameters (multipart/form-data):**
| Field | Type | Description |
|-------|------|-------------|
| `prompt` | string | Instruction to run on the file |
| `html_file` | file | HTML file to process |

**JSON response:**
```json
{
  "transaction_id": "abc123",
  "result": "...",
  "stderr": "...",
  "html": "<html>...</html>"
}
```

**curl example:**
```bash
curl -s -X POST http://localhost:8000/process \
  -F 'prompt=Translate all visible text to Swedish, change lang to sv. Save to index.html.' \
  -F 'html_file=@my_file.html;type=text/html' \
  | python3 -c "import sys,json; r=json.load(sys.stdin); open('output.html','w').write(r['html'])"
```

### `POST /process-repo`
Clone a Git repository over SSH, run OpenCode inside the repository, then optionally commit and push the changes.

**JSON body:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `repo_url` | string | yes | SSH Git URL, for example `git@github.com:owner/repo.git` |
| `ssh_private_key` | string | yes | Private SSH key with repository read/write access |
| `instruction` | string | yes | Instruction to execute in the repository |
| `branch` | string | no | Branch to clone. Defaults to the repository default branch |
| `push` | boolean | no | When `true`, commits and pushes changes back to the branch |
| `commit_message` | string | no | Commit message used when `push=true` |
| `git_user_name` | string | no | Commit author name |
| `git_user_email` | string | no | Commit author email |

**curl example:**
```bash
python3 - <<'PY' | curl -s -X POST http://localhost:8000/process-repo \
  -H 'Content-Type: application/json' \
  --data-binary @- | python3 -m json.tool
import json
from pathlib import Path

payload = {
    "repo_url": "git@github.com:owner/hugo-site.git",
    "branch": "main",
    "ssh_private_key": Path("deploy_key").read_text(),
    "instruction": "Refais le site Hugo avec un design plus moderne, puis vérifie que le build Hugo fonctionne.",
    "push": True,
    "commit_message": "Redesign Hugo site with OpenCode",
}
print(json.dumps(payload))
PY
```

The response includes `changed`, `status`, `diff_stat`, `diff`, OpenCode logs, and commit/push metadata when `push=true`.

## Tests

**Full test (Docker build + health + translation):**
```bash
python3 test_api.py
```

**Simple test (service already running):**
```bash
python3 test_simple.py
# Result saved to output.html
```

## `.env` configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `OVH_AI_KEY` | OVH AI API key (required) | — |
| `OVH_BASE_URL` | OVH AI endpoint | `https://oai.endpoints.kepler.ai.cloud.ovh.net/v1` |
| `OVH_MODEL` | Model to use | `Qwen2.5-Coder-32B-Instruct` |
| `OPENCODE_TIMEOUT` | OpenCode execution timeout in seconds | `600` |
| `GIT_CLONE_TIMEOUT` | Git clone timeout in seconds | `600` |
| `GIT_PUSH_TIMEOUT` | Git push timeout in seconds | `600` |
| `REPO_DIFF_MAX_CHARS` | Maximum diff size returned by `/process-repo` | `60000` |

## Architecture

```
OpenCodeAPIWrapper/
├── app/
│   ├── main.py          # FastAPI application
│   └── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── test.html            # Sample HTML file (FR calculator)
├── test_api.py          # Full test suite
└── test_simple.py       # Minimal test script
```

Each `/process` request creates an isolated temporary system user, runs OpenCode in their home directory, then deletes the user and all their files.
