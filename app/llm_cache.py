"""Proxy de cache LLM — limite le temps et les appels API pendant les tests.

OpenCode appelle une API LLM compatible OpenAI (OVH AI Endpoints). Ce proxy
local s'intercale DEVANT OVH : pour des requêtes ``/chat/completions`` (ou
``/completions``) IDENTIQUES, il renvoie la réponse mise en CACHE au lieu de
ré-appeler l'API distante — donc 0 appel réseau + réponse quasi instantanée
quand le prompt est le même (cf. tests e2e : prompt de génération de site
CONSTANT). En production les prompts varient par tenant → faible taux de hit,
inoffensif (simple passthrough).

Cache disque LRU borné (``LLM_CACHE_MAX_MB``, déf. 200 Mo). Clé = sha256 du
chemin + corps JSON canonicalisé (ordre des clés ignoré, champs volatils
``stream``/``stream_options`` ignorés). Les réponses en streaming (SSE) sont
bufferisées puis rejouées telles quelles. Désactivable via
``LLM_CACHE_ENABLED=false``.
"""
import base64
import hashlib
import json
import os
import threading
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

# Upstream réel (OVH). On accepte LLM_UPSTREAM_URL, sinon OVH_BASE_URL (même env
# que le wrapper), sinon l'endpoint OVH par défaut.
UPSTREAM = (os.environ.get("LLM_UPSTREAM_URL")
            or os.environ.get("OVH_BASE_URL")
            or "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1").rstrip("/")
API_KEY = os.environ.get("OVH_AI_KEY") or os.environ.get("OVH_API_KEY") or ""
CACHE_DIR = Path(os.environ.get("LLM_CACHE_DIR", "/cache/llm"))
MAX_BYTES = int(os.environ.get("LLM_CACHE_MAX_MB", "200")) * 1024 * 1024
ENABLED = os.environ.get("LLM_CACHE_ENABLED", "true").lower() in ("1", "true", "yes")
CACHEABLE_SUFFIXES = ("/chat/completions", "/completions")

CACHE_DIR.mkdir(parents=True, exist_ok=True)
_evict_lock = threading.Lock()

proxy = FastAPI(title="LLM cache proxy")


def _cache_key(path: str, body: bytes, auth: str = "") -> str:
    """Clé déterministe : chemin + corps JSON canonicalisé (les requêtes
    identiques au flag de streaming près partagent la même entrée).

    L'identité de l'appelant (clé API) entre dans la clé : les clés sont
    per-tenant, un cache partagé ne doit JAMAIS rejouer à un client la réponse
    générée pour un autre."""
    try:
        obj = json.loads(body)
        if isinstance(obj, dict):
            obj.pop("stream", None)
            obj.pop("stream_options", None)
        canon = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    except Exception:
        canon = body
    return hashlib.sha256(auth.encode() + b"\n" + path.encode() + b"\n" + canon).hexdigest()


def _split_upstream(path: str) -> tuple[str, str]:
    """Extrait un upstream encodé dans le chemin (`/u/<base64url>/v1/...`).

    Le wrapper y encode la passerelle IA MESURÉE de la plateforme quand la
    génération est facturée au crédit d'un tenant ; sans préfixe, l'upstream
    reste celui de l'environnement (OVH direct)."""
    if not path.startswith("/u/"):
        return UPSTREAM, path
    rest = path[3:]
    token, _, tail = rest.partition("/")
    try:
        target = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4)).decode()
    except Exception:
        return UPSTREAM, path
    if not target.startswith(("http://", "https://")):
        return UPSTREAM, path
    return target.rstrip("/"), "/" + tail


def _meta_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.meta"


def _body_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.body"


def _read_cache(key: str):
    mp, bp = _meta_path(key), _body_path(key)
    if not (mp.exists() and bp.exists()):
        return None
    try:
        meta = json.loads(mp.read_text())
        data = bp.read_bytes()
        now = time.time()
        os.utime(mp, (now, now))          # LRU : « touch » à chaque hit
        os.utime(bp, (now, now))
        return meta, data
    except Exception:
        return None


def _write_cache(key: str, status: int, content_type: str, data: bytes) -> None:
    try:
        _body_path(key).write_bytes(data)
        _meta_path(key).write_text(json.dumps(
            {"status": status, "content_type": content_type,
             "ts": int(time.time()), "size": len(data)}))
    except Exception:
        return
    _evict()


def _evict() -> None:
    """LRU : supprime les plus anciennes entrées tant que le total > MAX_BYTES."""
    with _evict_lock:
        bodies = sorted(CACHE_DIR.glob("*.body"), key=lambda p: p.stat().st_mtime)
        total = sum(p.stat().st_size for p in bodies)
        i = 0
        while total > MAX_BYTES and i < len(bodies):
            b = bodies[i]
            i += 1
            try:
                sz = b.stat().st_size
                b.unlink(missing_ok=True)
                _meta_path(b.stem).unlink(missing_ok=True)
                total -= sz
            except Exception:
                continue


@proxy.get("/health")
async def _health():
    return {"status": "ok", "cache": ENABLED, "dir": str(CACHE_DIR),
            "max_mb": MAX_BYTES // (1024 * 1024), "upstream": UPSTREAM}


@proxy.api_route("/{full_path:path}", methods=["GET", "POST"])
async def _forward(full_path: str, request: Request):
    path = "/" + full_path
    body = await request.body()
    # OpenCode tape <proxy>[/u/<b64>]/v1/... → on mappe sur l'upstream (qui finit
    # déjà par /v1), per-requête s'il est encodé dans le chemin, sinon l'env.
    upstream, path = _split_upstream(path)
    target = upstream + (path[len("/v1"):] if path.startswith("/v1") else path)

    # Clé d'API de l'appelant : conservée telle quelle quand elle est fournie (clé
    # IA du tenant → la passerelle mesure sa conso) ; sinon on injecte celle de
    # l'environnement (OVH direct).
    caller_auth = request.headers.get("authorization", "")

    cacheable = (request.method == "POST" and ENABLED
                 and any(path.endswith(s) for s in CACHEABLE_SUFFIXES))
    key = _cache_key(path, body, caller_auth) if cacheable else None
    if cacheable:
        hit = _read_cache(key)
        if hit:
            meta, cbody = hit
            return Response(content=cbody, status_code=meta.get("status", 200),
                            media_type=meta.get("content_type", "application/json"),
                            headers={"X-LLM-Cache": "HIT"})

    # Passthrough (MISS ou non-cacheable). On force l'identity (pas de gzip) pour
    # stocker/rejouer un corps non compressé, et on (ré)injecte la clé API OVH.
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host", "content-length", "accept-encoding")}
    if API_KEY and not caller_auth:
        headers["authorization"] = f"Bearer {API_KEY}"

    client = httpx.AsyncClient(timeout=httpx.Timeout(900.0))
    try:
        req = client.build_request(request.method, target, headers=headers, content=body)
        resp = await client.send(req, stream=True)
    except Exception as e:
        await client.aclose()
        return JSONResponse({"error": f"upstream unreachable: {e}"}, status_code=502)

    content_type = resp.headers.get("content-type", "application/json")
    status = resp.status_code
    chunks: list[bytes] = []
    # An interrupted SSE answer is not reusable: replaying it makes every
    # subsequent identical generation fail in exactly the same way.  Regular
    # JSON responses are complete once the HTTP response is complete; SSE
    # responses must explicitly end with the OpenAI ``[DONE]`` sentinel.
    is_sse = "text/event-stream" in content_type.lower()
    stream_complete = not is_sse
    sse_tail = b""

    async def _gen():
        nonlocal stream_complete, sse_tail
        try:
            async for chunk in resp.aiter_raw():
                chunks.append(chunk)
                if is_sse:
                    # A marker can span two TCP chunks, hence the small tail.
                    sse_tail = (sse_tail + chunk)[-32:]
                    if b"data: [DONE]" in sse_tail:
                        stream_complete = True
                yield chunk
        finally:
            await resp.aclose()
            await client.aclose()
            if cacheable and status == 200 and stream_complete:
                _write_cache(key, status, content_type, b"".join(chunks))

    return StreamingResponse(_gen(), status_code=status, media_type=content_type,
                             headers={"X-LLM-Cache": "MISS"})
