#!/usr/bin/env python3
import urllib.request
import urllib.error
import json
import sys

API_URL = "http://localhost:8000"
HTML_FILE = "test.html"
OUTPUT_FILE = "output.html"
PROMPT = (
    "Edit the file index.html and translate ALL its visible text to Swedish. "
    "Change the lang attribute to 'sv'. "
    "Do NOT touch HTML structure, CSS, or JavaScript logic. "
    "Save directly to index.html."
)

with open(HTML_FILE, "rb") as f:
    html_bytes = f.read()

boundary = "----Boundary"
body = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="prompt"\r\n\r\n'
    f"{PROMPT}\r\n"
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="html_file"; filename="index.html"\r\n'
    f"Content-Type: text/html\r\n\r\n"
).encode() + html_bytes + f"\r\n--{boundary}--\r\n".encode()

req = urllib.request.Request(
    f"{API_URL}/process",
    data=body,
    method="POST",
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
)

try:
    with urllib.request.urlopen(req, timeout=360) as resp:
        result = json.loads(resp.read())
        html = result.get("html", "")
        if not html.strip():
            print("Erreur : champ 'html' vide")
            print(result.get("stderr", "")[:500])
            sys.exit(1)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Sauvegardé dans {OUTPUT_FILE}")
except urllib.error.HTTPError as e:
    print(f"Erreur HTTP {e.code} : {e.read().decode()[:500]}")
    sys.exit(1)
