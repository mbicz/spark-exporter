# Native inference metrics and tokens per second

Spark Exporter observes inference servers that are already running. It does not
serve models and does not depend on a dashboard. The telemetry path is:

```text
engine's native Prometheus/OpenMetrics endpoint
    -> Spark Exporter local read-only metrics proxy (or a direct Prometheus scrape)
    -> Prometheus raw time series
    -> engine recording rules (5-minute counter rates)
    -> optional dashboard querying Prometheus
```

The package ships `prometheus/engine-rules.yml`, which defines two records:

- `spark:engine_input_tokens:rate5m`: input tokens per wall-clock second.
- `spark:engine_output_tokens:rate5m`: generated tokens per wall-clock second.

The installed copy is `/usr/share/spark-exporter/prometheus/engine-rules.yml`.
Any Prometheus installation can opt into it:

```yaml
rule_files:
  - /usr/share/spark-exporter/prometheus/engine-rules.yml
```

Or copy the file to a location Prometheus already owns. The package does not
edit another Prometheus installation's configuration, apply rules, or restart
it; the operator decides when to add and reload this configuration.

Scrape native metrics as `job="spark-inference"`, `cluster="spark"`,
`instance="<stable host identity>"`, and `engine_port="<local metrics port>"`.
Direct scrapes and proxy scrapes are alternatives for the same endpoint. Model
labels come from native samples, never from a guessed process name.

## What the token rate means

Output tok/s is the sum of generated-token counter rates. It is not GPU
utilization, not an average of per-request decode speeds, not a request rate,
not a draft-token rate, and not a histogram observation count. Each raw counter
passes through `rate(...[5m])` before summing, so Prometheus handles counter
resets individually. Two scrapes are needed before a rate exists, and a full
five minutes of samples gives a mature window. Native completion-accounted
counters advance when requests finish, so streaming bursts lag actual
generation.

Every record keeps `cluster`, `instance`, `engine_port`, the engine-family label
`engine`, and the native `model_name` when present. Triton's native `model` and
`version` become `model_name` and `model_version`. Native `engine_type` (SGLang
serving role or TensorRT backend) is kept when present. In native vLLM the
`engine` label is a worker ordinal, not the engine family: ordinal rates are
summed before the `engine="vllm"` record is attached. SGLang streaming and
custom-label request partitions are summed after rate calculation.

A missing token family produces no record. It is not zero, not a guessed speed,
and not a fallback from a gauge. A counter that stops increasing produces a
measured zero. A missing input family does not suppress a real output family.
Servers without native model labels stay unlabeled; they are not merged across
endpoints or hosts into an invented model. A Triton repository name such as
`tensorrt_llm` is a model identifier, not necessarily a Hugging Face checkpoint.

Only an explicit `prometheus_replica` label is treated as a duplicate
observation: `max without (prometheus_replica)` of rates deduplicates copies
with otherwise identical native labels. Independent model workers, hosts, ports,
versions and serving roles are not deduplicated. This is not a substitute for
storage-level HA deduplication, and encoding an independent worker as
`prometheus_replica` is wrong. Arbitrary native `replica` labels are not assumed
to be HA copies. Do not scrape one distributed worker set through several API
listeners and add the resulting endpoint records; endpoint identity cannot prove
that those listeners represent independent work.

Fleet or model totals are a downstream choice. Add records only after
establishing independent serving endpoints and a real shared model identity,
keeping the `engine` and model/version/role boundaries unless a deliberate
cross-engine comparison is intended. A model-less host total is not a model
total.

## Selected families

Every selected value below is cumulative in tokens; its 5-minute rate is
tokens/second. No alternate family is used as a fallback when the selected one
is missing.

| Engine label | Input family | Output family | Native identity/partition labels |
| --- | --- | --- | --- |
| `vllm` | `vllm:prompt_tokens_total` | `vllm:generation_tokens_total` | `model_name`, `engine` worker ordinal; definitions use Python `Counter` names without `_total`, exposition adds `_total` |
| `sglang` | `sglang:prompt_tokens_total` | `sglang:generation_tokens_total` | `model_name`, `engine_type`, `is_streaming`; optional priority/custom labels |
| `llamacpp` | `llamacpp:prompt_tokens_total` | `llamacpp:tokens_predicted_total` | Single-model server counters carry no model label |
| `tensorrt-llm` | `trtllm_prompt_tokens_total` | `trtllm_generation_tokens_total` | `model_name`, `engine_type`; counted on request completion |
| `triton-tensorrt-llm` | `nv_llm_input_token_len_sum{response_metric_type="total_input_tokens"}` | `nv_llm_output_token_len_sum{response_metric_type="total_output_tokens"}` | `model`, `version`, `response_metric_type`; cumulative histogram sums, not `_count` |
| `tensorfold` | `tensorfold:prompt_tokens_total` | `tensorfold:generation_tokens_total` | Native renderer has no model labels; counters describe finished requests |

Input semantics differ between engines. vLLM and SGLang count prefill/prompt
tokens; TensorFold counts prompt tokens of finished requests; llama.cpp counts
evaluated prompt work, which excludes cache reuse. Input rates are therefore not
directly comparable as identical work across engines.

TensorFold's aliases keep the `tensorfold:` prefix rather than a `vllm:` one.
They mirror queue, cache, latency and speculation readings, and they are not
additional generated tokens. Draft and live-stream counters stay raw and are not
added to finished output tokens.

## Endpoint defaults and operator toggles

These are facts for engine owners. The exporter does not unlock, restart or
reconfigure a running server.

| Server | Native endpoint and port | Engine-owner requirement |
| --- | --- | --- |
| vLLM OpenAI server | `/metrics`, 8000 | Exposed by a normal `vllm serve`; the core counters need no separate flag |
| SGLang | `/metrics`, 30000 | Start with `--enable-metrics`; disabled by default |
| llama.cpp | `/metrics`; binary default 9931, Docker examples set `--port 8080` | `--metrics` or `LLAMA_ARG_ENDPOINT_METRICS` enables the normally disabled endpoint. Router mode needs `?model=<id>`, which this fixed-path proxy does not discover |
| TensorRT-LLM `trtllm-serve` | `/prometheus/metrics`, 8000 | `return_perf_metrics: true` in the engine's YAML enables the collector and route. Counts are published on request completion |
| Triton with TensorRT-LLM backend | `/metrics`, metrics port 8002 | Metrics-enabled build with `--allow-metrics` left on, and a backend build exposing the token histograms. Generic Triton request/GPU metrics carry no token counts |
| TensorFold | `/metrics` (alias `/v1/metrics`), 8080 | Native metrics are always available. With API keys configured, `--metrics-open` is required for credential-free collection |
| Ollama | API on 11434, no passive metrics target | No cumulative native Prometheus token counter exists. Per-response `eval_count / eval_duration` is per-request speed, not wall-clock tok/s |

TensorRT native serving and the Triton backend are different implementations.
The native `/metrics` route is runtime-statistics JSON, and documented versions
consume iteration-statistics queue entries when it is polled. Use
`/prometheus/metrics` for TensorRT scrapes. JSON snapshots are never converted
into cumulative token counters.

## Discovery

Discovery tries `/prometheus/metrics` first. Before any generic `/metrics`
fallback it reads `/openapi.json` and refuses a route advertised as JSON
diagnostics or TensorRT native metrics. The TensorRT app registers `GET /metrics`
to `get_iteration_stats`, which returns a `JSONResponse`; the guard prevents a
disabled native Prometheus from causing a diagnostic-queue read. A
schema-advertised native Prometheus path is also a reason to skip generic
diagnostics. Auth failures and non-404 schema errors fail closed.

For Triton, `nv_trt_llm_v1_metrics{v1_specific_metric="total_generation_tokens"}`
and inflight-batcher readings are gauges, not cumulative output counters.
Neither those nor `nv_inference_request_success`, `nv_inference_count` or a
token-histogram `_count` is normalized to tok/s. The backend's `_sum` is
cumulative token observations; the `_count` counts requests.

## Discovery boundary

The worker range defaults to 8000–8031, augmented with the known ports 8080,
9931 and 30000 (Triton's 8002 is already inside the range). Only loopback
listeners in that bounded set are candidates. Other ports need explicit
configuration. There is no arbitrary remote proxy, no whole-host port sweep and
no Ollama polling.

The proxy tries `/prometheus/metrics` first, then guarded `/metrics`, and keeps
the successful endpoint for `/inference/<port>/metrics`. Explicit per-port
selection with `--engine-endpoint PORT:PATH` skips the fallback and accepts only
those two paths. `/v1/metrics` is a TensorFold alias, not a second scrape target.
Only valid Prometheus/OpenMetrics is relayed; names, labels and raw queue or
latency families are never rewritten. Discovery succeeds only on an actual
metrics response, not on a listening port or a process command line.

Collection never sends generation requests, restarts models, changes serving
flags, supplies API credentials or follows arbitrary proxy destinations.
Engines whose metrics are disabled, private or absent stay unavailable until
their owner changes them.

## Queue and latency stay native

The proxy preserves counters and every native queue, running-request, latency,
cache and histogram sample. The rule file adds only the two token-rate records,
so it cannot turn a latency average into a p95 or silently replace a raw gauge.

Representative raw families:

- vLLM: `vllm:num_requests_running`, `vllm:num_requests_waiting`, and its
  seconds-based TTFT, inter-token and end-to-end histograms.
- SGLang: `sglang:num_running_reqs`, `sglang:num_queue_reqs`, native latency
  histograms. `sglang:gen_throughput` is an instantaneous gauge, not the
  normalized rate.
- llama.cpp: `llamacpp:requests_processing`, `llamacpp:requests_deferred`, and
  average-throughput gauges.
- TensorRT native: request queue and latency families. Triton:
  `nv_trt_llm_request_metrics{request_type="waiting"}` and request-duration
  counters in microseconds.
- TensorFold: `tensorfold:requests_running`, `tensorfold:requests_waiting`,
  `tensorfold:generation_tokens_running`, KV occupancy per `pool`, and the
  `tensorfold:request_latency_seconds`, `time_to_first_token_seconds` and
  `request_time_per_output_token_seconds` histograms.

A percentile needs compatible native buckets, or a native summary quantile with
its own limitations. No cross-engine p95 is derived from averages, cumulative
duration divided by requests, or mixed units.

## Sources

Metric spellings, units, defaults and toggles were compared against these
official repositories. A different deployed version must be compared against its
own exposition before its accounting is trusted.

1. **vLLM** `ed1a7307950ab5e6593d25123579ed7ca998b350`: [counter definitions and worker labels](https://github.com/vllm-project/vllm/blob/ed1a7307950ab5e6593d25123579ed7ca998b350/vllm/v1/metrics/loggers.py#L499-L805), [endpoint and default port](https://github.com/vllm-project/vllm/blob/ed1a7307950ab5e6593d25123579ed7ca998b350/docs/usage/metrics.md#L1-L30).
2. **SGLang** `5884e4fec6c0f4453940605896154f5fba4badd9`: [tokenizer labels and counters](https://github.com/sgl-project/sglang/blob/5884e4fec6c0f4453940605896154f5fba4badd9/python/sglang/srt/observability/metrics_collector.py#L1804-L1864), [finished-request increments](https://github.com/sgl-project/sglang/blob/5884e4fec6c0f4453940605896154f5fba4badd9/python/sglang/srt/observability/metrics_collector.py#L2092-L2151), [default port](https://github.com/sgl-project/sglang/blob/5884e4fec6c0f4453940605896154f5fba4badd9/python/sglang/srt/arg_groups/fields/serving.py#L64-L66).
3. **llama.cpp** `7aca0d2ea8d92b56911233b822db01f9fbb1ae50`: [port and metrics flag](https://github.com/ggml-org/llama.cpp/blob/7aca0d2ea8d92b56911233b822db01f9fbb1ae50/tools/server/README.md#L193-L227), [Docker port examples](https://github.com/ggml-org/llama.cpp/blob/7aca0d2ea8d92b56911233b822db01f9fbb1ae50/tools/server/README.md#L454-L464), [metric types and router restriction](https://github.com/ggml-org/llama.cpp/blob/7aca0d2ea8d92b56911233b822db01f9fbb1ae50/tools/server/README.md#L1217-L1240), [native exposition and token accounting tests](https://github.com/ggml-org/llama.cpp/blob/7aca0d2ea8d92b56911233b822db01f9fbb1ae50/tools/server/tests/unit/test_metrics.py#L18-L123).
4. **TensorRT-LLM native** `c1d51b22b23b74dfa9fa4cee23fefc3cc315b159`: [token counter units and labels](https://github.com/NVIDIA/TensorRT-LLM/blob/c1d51b22b23b74dfa9fa4cee23fefc3cc315b159/tensorrt_llm/metrics/collector.py#L235-L245), [completion-only accounting](https://github.com/NVIDIA/TensorRT-LLM/blob/c1d51b22b23b74dfa9fa4cee23fefc3cc315b159/tensorrt_llm/metrics/collector.py#L650-L721), [opt-in and model labels](https://github.com/NVIDIA/TensorRT-LLM/blob/c1d51b22b23b74dfa9fa4cee23fefc3cc315b159/tensorrt_llm/serve/openai_server.py#L1006-L1014), [Prometheus route mount](https://github.com/NVIDIA/TensorRT-LLM/blob/c1d51b22b23b74dfa9fa4cee23fefc3cc315b159/tensorrt_llm/serve/openai_server.py#L1432-L1457).
5. **Triton TensorRT-LLM backend** `4a2a42a9f66e287c0eeb2dd12ade7fccef787506`: [backend metrics and gauge types](https://github.com/triton-inference-server/tensorrtllm_backend/blob/4a2a42a9f66e287c0eeb2dd12ade7fccef787506/README.md#L411-L475), with the backend source in submodule `ca88df0a400515b8206a19eb30cd047eb7661035`: [response_metric_type values](https://github.com/NVIDIA/TensorRT-LLM/blob/ca88df0a400515b8206a19eb30cd047eb7661035/triton_backend/inflight_batcher_llm/src/custom_metrics_reporter/custom_metrics_reporter.cc#L67-L71).
6. **TensorFold** `8c36359d533debdbb8f53a4229b0bb9362c9c783`: [endpoint, authentication and `--metrics-open`](https://github.com/ashhart/TensorFold/blob/8c36359d533debdbb8f53a4229b0bb9362c9c783/docs/api.md#L1-L18), [native Prometheus routes](https://github.com/ashhart/TensorFold/blob/8c36359d533debdbb8f53a4229b0bb9362c9c783/docs/api.md#L62-L67), [metrics semantics and prefixed aliases](https://github.com/ashhart/TensorFold/blob/8c36359d533debdbb8f53a4229b0bb9362c9c783/docs/api.md#L468-L490), [exact prefix](https://github.com/ashhart/TensorFold/blob/8c36359d533debdbb8f53a4229b0bb9362c9c783/zig/src/server/metrics.zig#L1-L6).
7. **Ollama** `b2137a1435033c7999385306ccdf33ec8a2a5733`: [route registrations, no `/metrics`](https://github.com/ollama/ollama/blob/b2137a1435033c7999385306ccdf33ec8a2a5733/server/routes.go#L2372-L2431), [per-response counts and durations](https://github.com/ollama/ollama/blob/b2137a1435033c7999385306ccdf33ec8a2a5733/api/types.go#L560-L570), [default API port](https://github.com/ollama/ollama/blob/b2137a1435033c7999385306ccdf33ec8a2a5733/envconfig/config.go#L20-L23).

## What to verify on a new host

Confirm the installed engine revision, its selected port and path, its metrics
toggles and network permissions; the presence of native counters and model
labels; completed, streamed and cancelled request accounting under normal
traffic; counter reset behaviour after an independently scheduled restart; and
whether multiple listeners expose independent workers or duplicate views of one
distributed worker set. TensorFold's native/Python differences, TensorRT's
response-path coverage and Triton's histogram availability must be checked
against the installed versions.
