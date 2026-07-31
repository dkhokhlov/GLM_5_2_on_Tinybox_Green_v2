#!/usr/bin/env python3
"""Measure prefill TPS and output (decode) TPS for the GLM-5.2 vLLM server.

Uses /v1/completions (raw prompt, no chat-template/parser overhead) with SSE
streaming so TTFT (prefill wall time) and decode rate are separated.
Cross-checks against vLLM's own /metrics.
"""
import http.client, json, time, sys

HOST, PORT, MODEL = "localhost", 8000, "glm-5.2-awq-int4"


def gen(prompt, max_tokens, temperature=0.0):
    body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": temperature, "stream": True,
            "stream_options": {"include_usage": True}, "ignore_eos": False}
    conn = http.client.HTTPConnection(HOST, PORT, timeout=600)
    t0 = time.perf_counter()
    conn.request("POST", "/v1/completions", body=json.dumps(body),
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    first = None
    usage = None
    for raw in resp:
        line = raw.decode("utf-8", "ignore").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            ev = json.loads(data)
        except Exception:
            continue
        if ev.get("usage"):
            usage = ev["usage"]
        ch = ev.get("choices") or []
        if ch and ch[0].get("text"):
            if first is None:
                first = time.perf_counter()
    tend = time.perf_counter()
    conn.close()
    if first is None:
        first = tend
    return t0, first, tend, usage


def _histogram(text, base):
    """Parse Prometheus histogram `base` from /metrics text.

    vLLM v0.26 exposes TTFT/TPOT as histograms (cumulative `_bucket{le=...}`
    plus `_sum`/`_count`), NOT summaries, so no `{quantile=...}` lines exist to
    print. Returns (buckets, count, sum): buckets = [(le, cum_count), ...] with
    finite `le` only, sorted. Returns None if the metric is absent.
    """
    buckets, count, sum_ = [], None, None
    for line in text.splitlines():
        if line.startswith(base + '_bucket{') and 'le="' in line:
            le = line.split('le="', 1)[1].split('"', 1)[0]
            if le != "+Inf":
                buckets.append((float(le), float(line.split()[-1])))
        elif line.startswith(base + "_count"):
            count = float(line.split()[-1])
        elif line.startswith(base + "_sum"):
            sum_ = float(line.split()[-1])
    if count is None:
        return None
    buckets.sort()
    return buckets, count, sum_


def _quantile(buckets, count, q):
    """histogram_quantile: linear-interpolate the q-th quantile across the
    cumulative `buckets` (PromQL semantics). Returns the last finite `le` when
    the quantile falls in the open +Inf bucket."""
    target = count * q
    prev_le, prev_cum = 0.0, 0.0
    for le, cum in buckets:
        if cum >= target:
            if cum > prev_cum:
                return prev_le + (target - prev_cum) / (cum - prev_cum) * (le - prev_le)
            return le
        prev_le, prev_cum = le, cum
    return prev_le


def metrics():
    try:
        c = http.client.HTTPConnection(HOST, PORT, timeout=10)
        c.request("GET", "/metrics")
        text = c.getresponse().read().decode()
        c.close()
        return {
            "ttft": _histogram(text, "vllm:time_to_first_token_seconds"),
            "tpot": _histogram(text, "vllm:request_time_per_output_token_seconds"),
        }
    except Exception as e:
        return {"err": str(e)}


def report(label, prompt, max_tokens):
    t0, first, tend, usage = gen(prompt, max_tokens)
    pt = usage["prompt_tokens"]
    ct = usage["completion_tokens"]
    ttft = first - t0
    gent = tend - first
    prefill_tps = pt / ttft if ttft > 0 else float("nan")
    output_tps = ct / gent if gent > 0 else float("nan")
    print(f"\n=== {label} ===")
    print(f"  prompt_tokens   = {pt}")
    print(f"  completion_tokens = {ct}")
    print(f"  TTFT (prefill wall) = {ttft*1000:.1f} ms")
    print(f"  prefill_tps     = {prefill_tps:8.1f} tok/s   (prompt_tokens / TTFT)")
    print(f"  gen_time        = {gent:.3f} s")
    if gent > 0.05:
        print(f"  output_tps      = {output_tps:8.1f} tok/s   (completion_tokens / gen_time)")
    else:
        print(f"  output_tps      = n/a   (prefill-only test; completion_tokens={ct})")


# Wait for the server. A cold start loads the 440 GB model (~6 min): a detached
# `make start` returns at once while the model loads in the background, so poll
# /health (HTTP 200 = ready) with a generous deadline and print progress.
DEADLINE = 720  # 12 min; cold load is ~6 min, leave headroom
print(f"waiting for server at {HOST}:{PORT} (cold model load takes ~6 min)...",
      flush=True)
_t0 = time.perf_counter()
_last_print = -1
ready = False
while time.perf_counter() - _t0 < DEADLINE:
    try:
        c = http.client.HTTPConnection(HOST, PORT, timeout=3)
        c.request("GET", "/health")
        resp = c.getresponse()
        resp.read()
        c.close()
        if resp.status == 200:
            ready = True
            break
    except Exception:
        pass
    elapsed = int(time.perf_counter() - _t0)
    if elapsed - _last_print >= 15:
        print(f"  ...waiting for server ({elapsed}s elapsed)", flush=True)
        _last_print = elapsed
    time.sleep(3)
if not ready:
    print(f"server not ready at {HOST}:{PORT} after {DEADLINE}s", file=sys.stderr)
    sys.exit(1)
print(f"server ready (waited {int(time.perf_counter() - _t0)}s)")

print("warmup (fills cudagraphs / kernels)...")
gen("Hello", 16)

SHORT = "Explain how ocean tides work, clearly and in detail. "
report("SHORT PROMPT", SHORT, 256)

# Meaningful prefill TPS needs a long prompt so prefill compute dominates
# overhead. A unique nonce prefix forces a COLD prefill every run: vLLM has
# prefix caching ON, so an identical repeat prompt is a cache hit (prefill
# skipped -> a bogus ~20000 tok/s). The nonce diverges the prefix from the
# start, so no cached blocks are reused.
LONG = (f"nonce-{time.perf_counter_ns()}\n"
        + "The quick brown fox jumps over the lazy dog. " * 600)  # ~6001 tok
report("LONG PROMPT (prefill-focused)", LONG, 8)

m = metrics()
print("\n=== vLLM /metrics (engine-side, all requests since server start) ===")
if "err" in m:
    print(f"  <unavailable: {m['err']}>")
else:
    for key, label in (("ttft", "TTFT"), ("tpot", "TPOT inter-token")):
        h = m.get(key)
        if not h:
            print(f"  {label}: <metric absent>")
            continue
        buckets, count, sum_ = h
        mean = sum_ / count if count else 0.0
        qs = {q: _quantile(buckets, count, q) for q in (0.5, 0.95, 0.99)}
        print(f"  {label:18} n={int(count):4d}  mean={mean*1000:8.1f}ms  "
              f"p50={qs[0.5]*1000:8.1f}ms  p95={qs[0.95]*1000:8.1f}ms  "
              f"p99={qs[0.99]*1000:8.1f}ms")
