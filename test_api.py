#!/usr/bin/env python3
"""
Script de test pour OpenCodeToAPI.
Lance docker-compose, attend que le service soit prêt, exécute les tests puis arrête le conteneur.
"""

import subprocess
import sys
import time
import urllib.request
import urllib.error
import json

PROJECT_DIR = "."
API_URL = "http://localhost:8000"
HTML_FILE = "test.html"
TIMEOUT_SECONDS = 120  # max temps d'attente pour que le service démarre


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=PROJECT_DIR, **kwargs)


def wait_for_service(url: str, timeout: int = TIMEOUT_SECONDS) -> bool:
    print(f"\n⏳ Attente du service sur {url}/health (max {timeout}s)...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=3) as resp:
                if resp.status == 200:
                    print("✅ Service prêt !")
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2)
    return False


def test_health() -> bool:
    print("\n--- TEST /health ---")
    try:
        with urllib.request.urlopen(f"{API_URL}/health", timeout=10) as resp:
            body = json.loads(resp.read())
            print(f"  Réponse : {body}")
            assert body.get("status") == "ok", f"Attendu 'ok', reçu : {body}"
            print("  ✅ PASS")
            return True
    except Exception as exc:
        print(f"  ❌ FAIL : {exc}")
        return False


OUTPUT_HTML_FILE = "test_translated.html"


def test_process() -> bool:
    print("\n--- TEST /process (traduction EN) ---")
    prompt = (
        "Edit the file index.html and translate ALL its visible text to English. "
        "Specifically you must change: "
        "the lang attribute from 'fr' to 'en', "
        "the <title> text, "
        "all heading and paragraph text, "
        "all input placeholder attributes, "
        "all button text labels, "
        "and every French string literal inside the JavaScript (e.g. 'Résultat : ', 'Division par zéro'). "
        "Do NOT touch the HTML structure, CSS rules, or JavaScript logic. "
        "Save the changes directly to index.html."
    )

    boundary = "----TestBoundary7a3f9e2b"
    body_parts = []

    # Champ texte "prompt"
    body_parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="prompt"\r\n\r\n'
        f"{prompt}\r\n"
    )

    # Fichier HTML
    with open(HTML_FILE, "rb") as fh:
        html_bytes = fh.read()

    body_parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="html_file"; filename="test.html"\r\n'
        f"Content-Type: text/html\r\n\r\n"
    )

    raw_body = (
        "".join(body_parts[:-1]).encode()
        + body_parts[-1].encode()
        + html_bytes
        + f"\r\n--{boundary}--\r\n".encode()
    )

    req = urllib.request.Request(
        f"{API_URL}/process",
        data=raw_body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )

    try:
        with urllib.request.urlopen(req, timeout=360) as resp:
            body = json.loads(resp.read())
            print(f"  transaction_id : {body.get('transaction_id')}")
            print(f"  --- stdout OpenCode ---\n{body.get('result', '')[:600]}")
            print(f"  --- stderr OpenCode ---\n{body.get('stderr', '')[:600]}")

            html_output = body.get("html") or ""

            if html_output.strip():
                with open(OUTPUT_HTML_FILE, "w", encoding="utf-8") as out:
                    out.write(html_output)
                print(f"  HTML traduit sauvegardé dans : {OUTPUT_HTML_FILE}")
                print(f"  Aperçu (300 chars) :\n{html_output[:300]}")
            else:
                print("  ⚠️  Champ 'html' vide — OpenCode n'a peut-être pas modifié le fichier.")
                raw = body.get("result", "")
                print(f"  stdout OpenCode (300 chars) :\n{raw[:300]}")

            print("  ✅ PASS")
            return bool(html_output.strip())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        print(f"  ❌ FAIL HTTP {exc.code} : {detail[:500]}")
        return False
    except Exception as exc:
        print(f"  ❌ FAIL : {exc}")
        return False


def _extract_html(text: str) -> str:
    """Extrait le bloc HTML d'une réponse qui peut contenir des balises markdown ```html ... ```."""
    import re
    # Cherche un bloc ```html ... ``` ou ``` ... ```
    m = re.search(r"```(?:html)?\s*(<!DOCTYPE.*?|<html.*?)</html\s*>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(0).split("\n", 1)[-1].rstrip("`").strip()
    # Cherche directement un <!DOCTYPE ou <html>
    m = re.search(r"(<!DOCTYPE\s+html.*?</html\s*>)", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1)
    return ""


def main():
    # 1. Build + démarrage du conteneur
    print("🔨 Build de l'image Docker...")
    result = run(["docker", "compose", "up", "--build", "-d"])
    if result.returncode != 0:
        print("❌ Échec du build/démarrage Docker.")
        sys.exit(1)

    # 2. Attente du service
    if not wait_for_service(API_URL):
        print("❌ Le service n'a pas démarré dans les temps.")
        run(["docker", "compose", "logs", "--tail=50"])
        run(["docker", "compose", "down"])
        sys.exit(1)

    # 3. Tests
    results = []
    results.append(test_health())
    results.append(test_process())

    # 4. Résumé
    passed = sum(results)
    total = len(results)
    print(f"\n{'='*40}")
    print(f"Résultats : {passed}/{total} tests passés")
    print(f"{'='*40}")

    # 5. Logs si échec
    if passed < total:
        print("\n📋 Logs du conteneur :")
        run(["docker", "compose", "logs", "--tail=100"])

    # 6. Arrêt du conteneur
    print("\n🛑 Arrêt du conteneur...")
    run(["docker", "compose", "down"])

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
