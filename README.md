# GLM-5.2-AWQ-INT4 on vLLM

[![GPU](https://img.shields.io/badge/GPU-4%C3%97_RTX_PRO_6000_(SM120)-blue)](https://www.techpowerup.com/gpu-specs/rtx-pro-6000-blackwell.c4230)
[![model](https://img.shields.io/badge/model-GLM--5.2--AWQ--INT4_(~743B_MoE)-success)](https://huggingface.co/cyankiwi/GLM-5.2-AWQ-INT4)
[![context](https://img.shields.io/badge/context-128k-brightgreen)](#performance)
[![quant](https://img.shields.io/badge/quant-compressed--tensors_INT4-orange)](#glm-52-architecture-whats-special--unusual)
[![TP](https://img.shields.io/badge/TP-4-blueviolet)](#quick-start)
[![license](https://img.shields.io/badge/license-MIT-yellow)](./LICENSE)

`cyankiwi/GLM-5.2-AWQ-INT4` (~743B MoE) · TP=4 · UVA expert offload · CUDA graphs · 4× RTX PRO 6000 (SM120)

## Performance

| Metric | Value | Notes |
|---|---|---|
| Decode | **26.6 tok/s** | 14→256 tokens, graphs ON (flat across context, DSA) |
| Prefill | 1232 tok/s | 6001→8 tokens |
| Decode (eager) | 8.6 tok/s | 14→256 tokens, graphs = **3.1×** speedup |

Target 10 tok/s — exceeded 2.66×.

## TOC

**Getting started**
- [Prerequisites](#prerequisites)
- [Security](#security)
- [Quick start](#quick-start)
- [Repository tree](#repository-tree)
- [Make targets](#make-targets)

**The stack**
- [File roles](#file-roles)
- [GLM-5.2 architecture](#glm-52-architecture-whats-special--unusual)
- [Image & capability tradeoffs](#image--capability-tradeoffs)
- [Capacity](#capacity-offload27-per-gpu)
- [Tuning (.env)](#tuning-env)

**Limits & operations**
- [Limits](#limits)
- [Gotchas](#gotchas)
- [Escape hatches](#escape-hatches)

## Prerequisites

- Docker Compose plugin + user in `docker` group (or `sudo`)
- NVIDIA Container Toolkit with the `nvidia` runtime registered
- 4× RTX PRO 6000 Blackwell (96 GiB each), each idle (<2 GiB in use)
- Python 3 (`make smoke`, `make bench`)

## Security

The server binds `0.0.0.0:8000` with no authentication. Restrict port 8000 to trusted clients (firewall or bind to a private interface).

## Quick start

| command | description |
|---|---|
| `make preflight` | read-only pre-launch checks |
| `make start` | start detached (~6 min, loads 440 GB) |
| `make logs` | tail logs → `Application startup complete` |
| `make smoke` | one-shot chat completion |
| `make bench` | prefill + decode TPS |

`make` (no target) → help.

## Repository tree

```
.
├── docker-compose.yml   # vllm service; flags resolved from .env
├── .env                 # all tunables (config of record; no secrets)
├── scripts/
│   ├── preflight.sh     # read-only pre-launch checks
│   └── bench.py         # prefill/decode TPS (streaming, stdlib only)
├── Makefile             # targets — see below (make help)
├── README.md
└── .gitignore           # bytecode/editor/OS; .env.*.local overrides
```

## Make targets

| Target | Action |
|---|---|
| `help` | list targets (default) |
| `preflight` | run `preflight.sh` |
| `pull` | pull image (~6 GB) |
| `start` | start detached (create + start) |
| `stop` | stop (keep container) |
| `restart` | stop then start |
| `logs` | tail server logs |
| `ps` | container status |
| `config` | render effective compose config |
| `health` | probe `/health` |
| `smoke` | one-shot chat completion |
| `bench` | prefill + decode TPS; waits for server (logs cold start) |

## File roles

| File | Role |
|---|---|
| `docker-compose.yml` | service `vllm` (`glm52-vllm`); RO-mounts HF cache at `/hf` (snapshot = symlinks → `blobs/`); `init: tini`; `restart: no`; flags via `${VAR}` from `.env` |
| `.env` | image tag, cache path, snapshot path, served name, tunables; auto-loaded by compose; tracked, override via `.env.*.local` |
| `scripts/preflight.sh` | exits non-zero on failure; checks compose plugin, docker group, `nvidia` runtime, 4 free GPUs (<2 GB used), model files, free RAM |
| `scripts/bench.py` | `/v1/completions` SSE streaming; warmup → SHORT (14→256, decode) → LONG (6001→8, prefill); decode = `completion_tokens / gen_time`; polls `/health` with progress logging (cold load ~6 min) |
| `Makefile` | `PORT` from `.env` (`HOST_PORT`); thin layer over `docker compose` |

## GLM-5.2 architecture (what's special / unusual)

| Aspect | Detail |
|---|---|
| `model_type` | `glm_moe_dsa` — a DeepSeek-V2 lineage model |
| Attention | **MLA** (Multi-head Latent Attention) — KV compressed to a ~576-dim latent per token/layer; **replicated, not sharded**, across TP ranks → tiny KV, long context feasible |
| **DSA** | DeepSeek Sparse Attention — sub-quadratic; decode cost stays nearly flat as context grows |
| MoE | 256 routed experts, **8 active/token**, + 1 **shared expert** (always on) |
| Layers / hidden | 78 / 6144 |
| Quant | `compressed-tensors` INT4, group-32, **asymmetric** (`symmetric:false`, `zp_dtype:int8`); repo named "AWQ" but `quant_method=compressed-tensors` |
| MoE backend | **Marlin** (forced — the WNA16 selector rejects TRTLLM for zero-point checkpoints; whitelist = triton/marlin/humming/trtllm/emulation) |
| Size | ~743B params; 440 GB on disk (83 shards); ~106 GiB weights/rank under TP=4 |

Consequences (why this matters for serving):

- **DSA → decode TPS barely drops with context.** Verified flat ~26.6 tok/s from 2K→32K. Speed is *not* the context ceiling; KV memory is.
- **MLA latent is replicated across TP ranks.** → 256K context needs ~13 GiB KV/rank → pinned-RAM / DCP problem (see Limits).
- **Only 8/256 experts active per token.** Small per-token compute; offloaded experts add PCIe fetch only if not resident.
- **Asymmetric INT4 zero-points.** Force Marlin and block TRTLLM / b12x / NVFP4 backends.

## Image & capability tradeoffs

| | |
|---|---|
| Tag | `vllm/vllm-openai:v0.26.0-cu129` |
| vLLM / FlashInfer / CUDA | 0.26.0 / 0.6.14 / 12.9 |
| Host driver | 580.173.02 |

**Why this tag:** the only release with **both** the SM120 sparse-MLA attention backend (`FLASHINFER_MLA_SPARSE_SM120`) and a FlashInfer (0.6.14) whose trtllm MLA decode kernel accepts `kv_scale_format`. No CUDA-13 release image exists for this model.

**Feature status for this checkpoint + SM120 stack:**

| Feature | Status | Why |
|---|---|---|
| MTP (speculative decode) | ❌ impossible | config has `num_nextn_predict_layers=1` but the AWQ-INT4 checkpoint **stripped all MTP weights**; fetching another model is out of scope |
| Expert swapping / prefetch offload | ❌ not stock | UVA is zero-copy on-demand (no decode-time expert pre-staging); predictive expert prefetch is research-only, not in vLLM for `glm_moe_dsa` |
| DCP (`--decode-context-parallel-size 4`) | ❌ blocked | would shard MLA KV (→ enable 256K) but the sparse-MLA DCP path rejects **fp8 KV** (we use `--kv-cache-dtype fp8`); no proven path for this stack. Upstream: [vllm#46514](https://github.com/vllm-project/vllm/pull/46514) (open) enables DCP on the `fp8_ds_mla` path for sparse MLA + MTP, stacked on merged [vllm#46076](https://github.com/vllm-project/vllm/pull/46076) |
| Expert parallel (`--enable-expert-parallel`) | ⚠️ null | tested: decode unchanged (8.6 = 8.6 eager); EP all-to-all cancels the residency gain → off |
| **UVA expert offload** | ✅ used | `--offload-backend uva --cpu-offload-gb 27 --cpu-offload-params experts` — experts in pinned host RAM, zero-copy PCIe |
| **CUDA graphs** | ✅ used | `--enforce-eager` OFF → **3.1× decode** (8.6→26.6); captured for bs ∈ {1,2} |
| Marlin WNA16 MoE | ✅ used | asymmetric INT4 via `compressed-tensors` |

## Capacity (offload=27, per GPU)

| Item | Size |
|---|---|
| Weights resident | 79 GiB |
| Offloaded → host (UVA) | 27 GiB |
| KV cache | 6.6 GiB |
| CUDA graphs | 0.04 GiB |
| VRAM budget (`util=0.95`) | 90 GiB |
| Pinned host RAM (4 ranks) | 108 GiB of 188 GB |

Decode is **framework-overhead-bound, not PCIe-bound** (most experts resident). Graphs remove per-token launch overhead across 100+ layers at bs=1. `max-num-seqs=1` → fixed batch → ideal capture (bs {1,2}).

## Tuning (`.env`)

| Knob | Default | Effect |
|---|---|---|
| `CPU_OFFLOAD_GB` | 27 | 27 → 128K pool (~150K tokens, pinned 108 GiB); lower (24) → faster decode but ~88K pool; floor ~22 |
| `EAGER_FLAG` | (empty) | empty = graphs ON (3.1× decode); `--enforce-eager` = debug only |
| `ESTIMATE_CUDAGRAPHS` | -1 | auto-reserve graph mem; `0` only with eager |
| `EP_FLAG` | (empty) | `--enable-expert-parallel` = NULL (all-to-all cancels gain); leave empty |
| `MAX_NUM_SEQS` | 1 | single-stream; do not raise |
| `MAX_MODEL_LEN` | 131072 | 128K serving; 256K blocked (see Limits) |
| `GPU_MEM_UTIL` | 0.95 | lower to 0.90 if OOM |

## Limits

- **256K context — blocked.** The 256K attempt needs `CPU_OFFLOAD_GB=33` → pinned 132 GiB; the 4-rank load transient thrashed/OOM-locked the box (proven-safe is offload=24/pinned 96; serving uses offload=27/pinned 108 for 128K). DCP=4 would shard MLA KV but the SM120 sparse-MLA DCP path rejects **fp8 KV**; no proven path (upstream: see the DCP row in [Feature status](#image--capability-tradeoffs)).
- **MTP — unavailable.** Config has `num_nextn_predict_layers=1` but checkpoint stripped MTP weights; no re-fetch.

## Gotchas

- **`content: null` is normal mid-reasoning.** GLM-5.2 is a reasoning model: chain-of-thought goes in `message.reasoning`; `content` stays `null` until thinking ends. A short `max_tokens` → `finish_reason: "length"` + `content: null` (looks empty, isn't a failure). Budget ≥512 tokens for trivial prompts; read `message.reasoning` for the thinking. (`make smoke` uses 1024 for this reason.)
- **Chat vs completions differ.** `bench.py` hits `/v1/completions` (raw, no reasoning parser); `/v1/chat/completions` applies the chat template + reasoning parser — token counts and latency are not directly comparable.

## Escape hatches

| Symptom | Fix |
|---|---|
| Graph capture error | `EAGER_FLAG=--enforce-eager`, `ESTIMATE_CUDAGRAPHS=0` |
| Marlin/UVA INT4 crash | `VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1` (not in Compose `environment:` — add it before use) |
| "Driver too old" | host driver ≥ image CUDA minor (580 ≥ cu12.9) |