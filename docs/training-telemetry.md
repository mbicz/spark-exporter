# Training telemetry: research note

Status: researched, not implemented. This note records what each training stack
actually exposes so a future collector starts from verified facts rather than
assumption. Nothing here is a claim about a running Spark host; no training
framework was installed on either host when this was written.

## What "training metrics" can mean

Three distinct sources exist, and only the first is passive and standard:

1. **A Prometheus endpoint the framework already serves.** Cumulative counters
   (steps, samples, tokens) and gauges (loss, learning rate, epoch). This is the
   only category that fits the exporter's existing read-only, bounded-loopback
   contract.
2. **A JSON or SSE progress API.** Real numbers, but not Prometheus text, and it
   usually requires an authenticated session or a running job handle.
3. **A local log or event file.** TensorBoard event files, `trainer_state.json`,
   stdout. Requires parsing a format that changes between library versions and
   belongs to a log shipper, not a scrape-time exporter.

## Verified framework inventory

### Ray Train

Serves Prometheus text from the dashboard agent, not from the trainer process.
The agent listens on `--dashboard-agent-listen-port` (default 52365) and exports
on `--metrics-export-port` (random unless pinned). A pinned release documents
exactly four families, all labelled `ray_train_run_name` and `ray_train_run_id`:

| Family | Type | Meaning |
|---|---|---|
| `ray_train_controller_state` | gauge, string state in a label | Controller state |
| `ray_train_worker_group_start_total_time_s` | counter, seconds | Worker group startup |
| `ray_train_worker_group_shutdown_total_time_s` | counter, seconds | Worker group shutdown |
| `ray_train_report_total_blocked_time_s` | counter, seconds | Checkpoint reporting, adds `ray_train_worker_world_rank` and `ray_train_worker_actor_id` |

Ray also serves HTTP service discovery at `/api/prometheus/sd` on the dashboard
port (8265 by default). There are no per-step loss, learning-rate or epoch
families: Ray Train v2 exposes four metrics, and Ray Data separately exposes
roughly fifty. A collector must therefore not promise loss curves from Ray.

Primary source: <https://docs.ray.io/en/latest/train/user-guides/monitor-your-application.html>
and <https://docs.ray.io/en/latest/ray-core/configure.html>.

### Kubeflow Training Operator (v1) and Kubeflow Trainer (v2)

The v1 operator serves `/metrics` on port 8080 by default, enabled without extra
configuration, with five job-lifecycle counters labelled `namespace` and
`framework`:

`training_operator_jobs_created_total`, `training_operator_jobs_deleted_total`,
`training_operator_jobs_successful_total`, `training_operator_jobs_failed_total`,
`training_operator_jobs_restarted_total`.

These count jobs, not training progress. Trainer v2 adds progress reporting, but
it is a Kubernetes API write path: the training script pushes
`progressPercentage`, `estimatedRemainingSeconds` and named metric pairs into
`TrainJob.status.trainerStatus` through the SDK or the controller's HTTPS
endpoint. Reading it requires a kubeconfig and cluster RBAC, which is outside a
host-local exporter's remit; a cluster-side Prometheus integration owns it.

Primary sources: <https://trainer.kubeflow.org/en/latest/legacy-v1/user-guides/monitoring.html>
and <https://trainer.kubeflow.org/en/latest/user-guides/trainjob-progress.html>.

### Determined AI

Exposes three separate scrape paths on the master, and the documented
configuration pins them explicitly:

- `/debug/prom/metrics` — master API server metrics, bearer token required.
- `prom/det-state-metrics` — per-task state metrics; the documented config
  extracts `experiment_id` from the `task_actor` label with a relabel rule.
- `/prom/det-http-sd-config` — HTTP service discovery for per-task
  cAdvisor/DCGM targets.

Every path requires an authorization token. The exporter never forwards
credentials, so Determined is only collectable when its operator publishes an
unauthenticated path or a sidecar performs the authenticated scrape.

Primary source: <https://github.com/determined-ai/works-with-determined/blob/main/observability/prometheus/prometheus.yml>.

### MLflow

Serves `/metrics` when started with `--expose-prometheus <dir>` or
`MLFLOW_EXPOSE_PROMETHEUS`. The families are HTTP request counters and latency
histograms for the tracking server, not training loss or step counters. Useful
for tracking-server health; not a training-progress source.

Primary sources: <https://mlflow.org/docs/latest/cli.html> and the Helm chart's
ServiceMonitor example.

### Unsloth

Unsloth is a training library, not a server. Unsloth Studio, the browser GUI,
exposes JSON rather than Prometheus text: `GET /api/train/metrics` returns loss,
learning rate and step, and `GET /api/train/stream` is an SSE progress stream.
Both require the Studio session's password, and the payload shape is a GUI
contract rather than a documented metrics schema. A training run driven from a
notebook or the `unsloth` CLI exposes nothing scrapeable at all.

Primary source: <https://unsloth.ai/docs/new/studio/start>.

### PyTorch Lightning, DeepSpeed, Accelerate, `torch.distributed`

None of these ship a Prometheus endpoint. Lightning logs through
`Logger`/`Callback` implementations (TensorBoard, CSV, W&B); DeepSpeed writes
its own logs; NCCL counters live in debug logs. Instrumenting them means adding
code to the training job or running an OpenTelemetry-style bridge, which is a
change to the workload, not a passive observation of it.

## Design consequences

A collector that follows the existing contract can support category 1 only, and
must discover the framework from the families it finds rather than from a
process name:

- Candidate ports: Ray metrics export (unpinned, so operator-supplied), Ray
  dashboard 8265, Kubeflow operator 8080, MLflow 5000. All collide with
  inference ports 8080 and the worker range, so an explicit
  `--training-endpoint PORT[:PATH]` opt-in is required rather than port scanning
  by default.
- Framework detection from family prefixes: `ray_train_`, `training_operator_`,
  `mlflow_`. Determined needs a path and a token, so it stays operator-supplied
  or unsupported.
- Normalized output: `spark_training_endpoint_up{framework}`, plus
  `spark_training_steps_total`, `spark_training_samples_total` and
  `spark_training_loss` only where a real cumulative counter or gauge exists.
  Ray and Kubeflow expose no step or loss family, so those series must stay
  absent for them rather than being synthesized.
- Raw preservation: relay the framework's own families unchanged, exactly as
  the inference proxy does, so nothing is lost when normalization does not
  apply.

## Open questions

- Whether a pinned Ray release exposes per-run step counters under a different
  prefix in `ray_train_*`; the four-family table is from a specific release and
  a different version must be checked against its own exposition.
- Whether Trainer v2 exposes the progress values through a Prometheus endpoint
  or only the Kubernetes API.
- Whether any MLflow build exposes training-run metrics (as opposed to tracking
  server HTTP metrics).
