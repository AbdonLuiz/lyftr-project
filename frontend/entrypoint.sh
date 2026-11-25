#!/bin/sh
set -eu
API_BASE_URL="${API_BASE_URL:-http://localhost:8000}"
escaped=$(printf '%s' "$API_BASE_URL" | sed 's/\\/\\\\/g; s/"/\\"/g')
printf 'window.__APP_CONFIG__ = { apiBaseUrl: "%s" };\n' "$escaped" > /usr/share/nginx/html/config.js
exec nginx -g 'daemon off;'
