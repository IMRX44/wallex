#!/usr/bin/env bash
# Deploy Pishro to Liara. Requires the Liara CLI (npm i -g @liara/cli) and, for non-interactive use,
#   LIARA_API_TOKEN  - Liara API token (Console > API)        LIARA_TEAM_ID - team id (optional)
#   APP              - unique app name (default: pishro)
# Secrets (WALLEX_API_KEY, LLM_*, DASHBOARD_PASSWORD, ...) are read from .env if present;
# LIARA_* deploy credentials are never copied into the app's environment, and PISHRO_MODE /
# LIVE_CONFIRM are left as they are on the app (switch them only with local/liara_live.cmd).
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
  liara create --app "$APP" --platform docker --plan "${PLAN:-medium-g2}" --feature-plan "${FEATURE_PLAN:-basic}" --network "$NETWORK" "${AUTH[@]}"
fi
# the CLI has no "disk list"; a failing create on redeploy means the disk exists. If it really does not,
# "liara deploy --disks data:/app/data" below fails loudly.
liara disk create --app "$APP" --name data --size "${DISK_GB:-2}" "${AUTH[@]}" 2>/dev/null   || echo ">> disk 'data' not created (already exists?) - deploy will verify the mount"

if [[ -f .env ]]; then
  echo ">> setting environment variables from .env"
  mapfile -t ENVS < <(grep -vE '^\s*(#|$)' .env | sed 's/\s*#.*$//' | grep '=' | grep -v '=$' | grep -vE '^(LIARA_|PISHRO_MODE=|LIVE_CONFIRM=)')
  [[ ${#ENVS[@]} -gt 0 ]] && liara env set --app "$APP" "${ENVS[@]}" --force "${AUTH[@]}"
fi

sed -i "s/\"app\": \"[^\"]*\"/\"app\": \"$APP\"/" liara.json
liara deploy --app "$APP" --port 8000 --disks data:/app/data --detach "${AUTH[@]}"
echo ">> deployed: https://$APP.liara.run"
