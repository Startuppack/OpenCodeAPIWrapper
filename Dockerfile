# ─── Stage 1 : téléchargement d'OpenCode ────────────────────────────────────
FROM debian:bookworm-slim AS opencode-downloader

ARG OPENCODE_VERSION=1.4.8

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        tar \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    MACHINE_ARCH=$(uname -m); \
    case "${MACHINE_ARCH}" in \
      x86_64)          ARCH="x64" ;; \
      aarch64|arm64)   ARCH="arm64" ;; \
      *) echo "Architecture non supportée: ${MACHINE_ARCH}" && exit 1 ;; \
    esac; \
    curl -fsSL \
      "https://github.com/anomalyco/opencode/releases/download/v${OPENCODE_VERSION}/opencode-linux-${ARCH}.tar.gz" \
      -o /tmp/opencode.tar.gz \
    && tar -xzf /tmp/opencode.tar.gz -C /tmp \
    && mv /tmp/opencode /usr/local/bin/opencode \
    && chmod +x /usr/local/bin/opencode \
    && rm /tmp/opencode.tar.gz

# ─── Stage 2 : image finale ──────────────────────────────────────────────────
FROM python:3.12-slim

# Outils système nécessaires pour gérer les utilisateurs
RUN apt-get update && apt-get install -y --no-install-recommends \
        passwd \
        bash \
        ca-certificates \
        curl \
        git \
        openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Copie du binaire OpenCode depuis le stage précédent
COPY --from=opencode-downloader /usr/local/bin/opencode /usr/local/bin/opencode

WORKDIR /app

# Dépendances Python
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code de l'application
COPY app/ .

# L'API doit tourner en root pour pouvoir créer/supprimer des utilisateurs
EXPOSE 8000

# start.sh lance le proxy de cache LLM (localhost:8011) puis l'API wrapper (8000).
CMD ["sh", "start.sh"]
