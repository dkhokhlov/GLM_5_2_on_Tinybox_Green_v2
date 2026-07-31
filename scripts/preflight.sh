#!/usr/bin/env bash
# Preflight checks for the GLM-5.2 vLLM deployment. Read-only; makes no changes.
# Run before: make start.
set -u

warn()  { printf '  WARN  %s\n' "$*"; }
fail()  { printf '  FAIL  %s\n' "$*"; FAILED=1; }
ok()    { printf '  ok    %s\n' "$*"; }
FAILED=0

echo "== docker =="
if docker compose version >/dev/null 2>&1; then
  ok "docker compose plugin present"
elif command -v docker-compose >/dev/null 2>&1; then
  warn "docker compose plugin missing, but legacy docker-compose binary found"
else
  fail "docker compose missing. Install it: sudo apt-get install docker-compose-plugin"
fi
if id -nG | grep -qw docker; then
  ok "user is in the docker group"
else
  warn "user not in docker group. Run docker with sudo, or: sudo usermod -aG docker \$USER (then re-login)"
fi

echo "== NVIDIA runtime =="
if docker info 2>/dev/null | grep -qi 'nvidia'; then
  ok "nvidia runtime detected in docker info"
else
  warn "could not confirm nvidia runtime (docker info may need sudo). Verify nvidia-container-toolkit is installed"
fi

echo "== GPUs (must be free for the full 96 GB to be usable) =="
if command -v nvidia-smi >/dev/null 2>&1; then
  ngpu=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
  maxused=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -nr | head -1)
  printf '  %s GPUs detected; busiest GPU has %s MiB in use\n' "$ngpu" "${maxused:-?}"
  if [ "${ngpu:-0}" -lt 4 ]; then fail "fewer than 4 GPUs visible (tensor-parallel-size=4 needs 4)"; fi
  if [ "${maxused:-99999}" -gt 2000 ]; then
    fail "a GPU has ${maxused} MiB in use. Free all GPUs first (other workloads occupy memory that vLLM needs)."
  else
    ok "all GPUs effectively idle"
  fi
else
  fail "nvidia-smi not found"
fi

echo "== model files =="
SNAP="/raid/huggingface_pavel/models--cyankiwi--GLM-5.2-AWQ-INT4/snapshots/0baf1f00d05d19e6e01e5451ed6cc54a0a93c5f2"
for f in config.json model.safetensors.index.json tokenizer_config.json tokenizer.json chat_template.jinja; do
  if [ -e "$SNAP/$f" ]; then ok "found $f"; else fail "missing $f"; fi
done

echo "== host RAM (pinned UVA offload needs free RAM) =="
avail=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)  # KiB
avail_gb=$(awk "BEGIN{printf \"%.0f\", $avail/1024/1024}")
printf '  %s GB RAM available\n' "$avail_gb"
if [ "${avail_gb:-0}" -lt 130 ]; then warn "less than 130 GB free; offload at CPU_OFFLOAD_GB=30 (needs ~120 GB pinned) may be tight"; fi

echo
if [ "$FAILED" -eq 0 ]; then printf 'PREFLIGHT: PASS\n'; else printf 'PREFLIGHT: FAILED (resolve the items above)\n'; fi
exit "$FAILED"
