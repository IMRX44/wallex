#!/usr/bin/env bash
# Deploy Pishro to Liara. Requires the Liara CLI (npm i -g @liara/cli) and, for non-interactive use,
#   LIARA_API_TOKEN  - Liara API token (Console > API)        LIARA_TEAM_ID - team id (optional)
#   APP              - unique app name (default: pishro)
# Secrets (WALLEX_API_KEY, LLM_*, DASHBOARD_PASSWORD, ...) are read from .env if present;
# LIARA_* deploy credentials are never copied into the app's environment.
set -euo pipefail
cd "$(dirname "$0")/.."
APP="${APP:-pishro}"
AUTH=()
[[ -n "${LIARA_API_TOKEN:-}" ]] && AUTH+=(--api-token "$LIARA_API_TOKEN")
[[ -n "${LIARA_TEAM_ID:-}" ]] && AUTH+=(--team-id "$LIARA_TEAM_ID")

NETWORK="${NETWORK:-$APP-net}"
if ! liara app list "${AUTH[@]}" 2>/dev/null | grep -qw "$APP"; then
  # non-interactive: the CLI otherwise prompts for a network
  if ! liara network list "${AUTH[@]}" 2>/dev/null | grep -qw "$NETWORK"; then
    echo ">> creating network $NETWORK"
    liara network create --name "$NETWORK" "${AUTH[@]}"
  fi
  echo ">> creating app $APP"
  liara create --app "$APP" --platform docker --plan "${PLAN:-medium-g2}" --network "$NETWORK" "${AUTH[@]}"
fi
if liara disk list --app "$APP" "${AUTH[@]}" 2>/dev/null | grep -qw data; then
  echo ">> disk 'data' exists"
else
  liara disk create --app "$APP" --name data --size "${DISK_GB:-2}" "${AUTH[@]}"
fi

if [[ -f .env ]]; then
  echo ">> setting environment variables from .env"
  mapfile -t ENVS < <(grep -vE '^\s*(#|$)' .env | sed 's/\s*#.*$//' | grep '=' | grep -v '=$' | grep -vE '^LIARA_')
  [[ ${#ENVS[@]} -gt 0 ]] && liara env set --app "$APP" "${ENVS[@]}" --force "${AUTH[@]}"
fi

sed -i "s/\"app\": \"[^\"]*\"/\"app\": \"$APP\"/" liara.json
liara deploy --app "$APP" --port 8000 --disks data:/app/data --detach "${AUTH[@]}"
echo ">> deployed: https://$APP.liara.run"
