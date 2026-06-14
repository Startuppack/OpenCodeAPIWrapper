#!/bin/sh
set -e
# Démarre le proxy de cache LLM (limite le temps + les appels API en rejouant
# depuis le cache les prompts identiques), puis l'API wrapper. Le proxy n'écoute
# que sur localhost : seul OpenCode (dans ce même conteneur) le tape.
if [ "${LLM_CACHE_ENABLED:-true}" != "false" ]; then
  uvicorn llm_cache:proxy --host 127.0.0.1 --port "${LLM_CACHE_PORT:-8011}" --log-level warning &
fi
exec uvicorn main:app --host 0.0.0.0 --port 8000
