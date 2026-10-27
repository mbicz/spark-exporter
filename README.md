# Spark Exporter

Spark Exporter publishes the telemetry a GB10 host already exposes, as
Prometheus text on port 9400. The GB10 is one machine sold under several names —
DGX Spark from NVIDIA, GX10 from ASUS, and rebadges from other vendors — with
identical silicon, so one exporter covers them all. The vendor datasheet
describes the hardware; see
[NVIDIA's DGX Spark specifications](https://www.nvidia.com/en-us/products/workstations/dgx-spark/#m-specs).
This project covers the software side: which sources the platform exposes, and
what they can tell you.

One Debian package installs everything. A pinned upstream node_exporter supplies
the kernel's own counters, a small Python adapter supplies the readings
node_exporter cannot reach, and both run as unprivileged systemd services. The
adapter merges the node counters into its own `/metrics` response, so a single
scrape returns the whole picture.

Readings come from four kinds of source. The driver publishes GPU telemetry
through `nvidia-smi`. The kernel publishes thermal zones, per-chip hwmon sensors,
per-policy CPU frequency, NVMe identity, interface state, RDMA counters and raw
`ethtool -S` driver statistics. The storage stack publishes block devices and USB
topology through sysfs. Inference engines publish their own Prometheus endpoints,
whose contents depend on the engine rather than on the host. The exporter reads
those sources and nothing else: it never writes to them, loads a driver or
changes a device setting.

This is a pre-1.0 release. The collectors and metric families described below
work and are verified on real GB10 hardware, but names and semantics may change
before 1.0. Pin a version if you depend on them.

## What it reports

**GPU.** Temperature and its limit, memory temperature, power draw against its
limits, every clock domain and its ceiling, PCIe generation and width, replay and
ECC counters, the active throttle reasons and the time the driver has spent
throttled for each one, per-process memory, and driver/CUDA/VBIOS identity.

**Host.** Memory pressure, per-zone thermal readings, per-chip hwmon sensors with
their critical and maximum limits, per-policy CPU frequency and governor, and
NVMe controller identity.

**Fabric.** Negotiated link speed, carrier, MTU, duplex, operational state and
PCIe width for every physical interface; kernel error, drop, collision and CRC
counters; raw RDMA state, rate and every readable counter with its unit; and the
complete `ethtool -S` driver statistics, including per-priority pause, CNP and
ECN counters.

**Storage.** Every physical block device with its transport, size, rotation and
the identity attributes the kernel publishes; every USB device with its port
path, vendor and product identifiers, and negotiated speed; and every mount on
USB or removable media, tagged with its transport.

**Inference.** A bounded, read-only proxy for the metrics an engine already
serves, plus normalized token rates, queue depth, cache occupancy, latency means
and speculative-decode acceptance.

**Model supervisor.** Optional lane accounting: readiness, leases, declared
memory, admission budget, port, priority, elapsed time and fences.

## What a reading means

Each `spark_*` family states one reading from one source. The unit is in the
family name, so `spark_gpu_temperature_celsius` is degrees Celsius and
`spark_gpu_clock_sm_hertz` is hertz. Fabric counters keep their native unit in a
`unit` label: `port_rcv_data` counts four-byte words and is labelled
`unit="4_bytes"`, while driver pause counters are gauges labelled `unit="raw"`.

A family appears in a scrape only when its source produced a value. The GB10
driver returns `N/A` for framebuffer size, memory clocks and power limits, so
those samples are missing from a GB10 host's output while
`spark_gpu_field_supported{field="memory_total_bytes"}` is 0. A reading of 0 means
the source reported 0, which is how the driver describes an idle GPU.

Rates belong to Prometheus, which sees consecutive scrapes. The exporter reports
the cumulative counter and leaves `rate()` to the query.

## Install

```sh
sudo apt install ./spark-exporter_0.1.3-1_arm64.deb   # or _amd64.deb on x86-64
sudo systemctl enable --now spark-exporter.service
curl --fail http://127.0.0.1:9400/metrics
```

The package depends on Python 3.9+, `ethtool` and `systemd`; the adapter uses
only the Python standard library. GPU readings come from the `nvidia-smi` that
ships with your installed driver, and the package never installs, replaces or
restarts a driver. Without a driver the GPU collector reports failure and every
other collector keeps working.

`spark-exporter.service` listens on `127.0.0.1:9400`.
`spark-node-exporter.service` listens on `127.0.0.1:19100` and is reachable only
from the adapter on the same host.

## Configure

Edit `/etc/default/spark-exporter`. The default line is:

```sh
SPARK_EXPORTER_ARGS="--listen-address 127.0.0.1 --node-exporter-url http://127.0.0.1:19100/metrics --enable-inference"
```

To serve the host's Tailscale address and read the model supervisor's lane
accounting:

```sh
SPARK_EXPORTER_ARGS="--listen-address tailscale --node-exporter-url http://127.0.0.1:19100/metrics --enable-supervisor --enable-inference"
```

`tailscale ip -4` is resolved once at startup, and a concrete IPv4 literal works
too. The listener must be a single address the kernel permits binding. Restart
the service after editing:

```sh
sudo systemctl restart spark-exporter.service
```

Endpoints carry no authentication. Keep the loopback default, or restrict port
9400 to your collector with a firewall or tailnet policy. Process IDs, model
names, GPU identifiers and storage serial numbers appear in the output.

## Endpoints

`GET /metrics` returns hardware, fabric and engine statistics in Prometheus text.
`spark_host_info{architecture=...,version=...} 1` identifies the exporter, with
`architecture` taken from the kernel's `platform.machine()`.

`GET /targets` returns a Prometheus HTTP service-discovery array for the
inference endpoints that currently answer with valid metrics. It is empty when
inference discovery is off. Targets point at this exporter's own address and port
9400, with the upstream engine reached through a loopback proxy:

```json
[{"targets":["100.100.1.2:9400"],"labels":{"__metrics_path__":"/inference/8007/metrics","engine_port":"8007"}}]
```

`GET /inference/<port>/metrics` fetches the engine's own metrics document from
loopback and returns it unchanged, including histogram buckets, exemplars and
model labels. The first scrape after the exporter starts has no rate yet; the
second one does.

Discovery looks at the bounded port range 8000–8031 plus the common engine ports
8080, 9931 and 30000, and at `PORT=` assignments in
`/etc/spark-exporter/lanes.d/*.conf` when supervisor integration is enabled.
Configuration files are read, never executed. `--engine-endpoint PORT:PATH` pins
one port to `/metrics` or `/prometheus/metrics` for engines whose OpenAPI schema
cannot be inspected.

TensorRT-LLM needs `return_perf_metrics: true` for `/prometheus/metrics` to
exist. Its plain `/metrics` is a JSON diagnostics endpoint that consumes
iteration-statistics queue entries when polled, so the exporter reads
`/openapi.json` before falling back and refuses a route advertised as JSON
diagnostics. Authenticated engines must expose metrics without a key, for example
TensorFold's `--metrics-open`; the exporter never forwards credentials.

## Inference statistics

With inference discovery enabled, `/metrics` carries a normalized view of each
engine endpoint alongside the raw native families:

| Metric | Meaning |
|---|---|
| `spark_engine_tokens_total{direction}` | Native cumulative prompt or generation tokens |
| `spark_engine_tokens_per_second{direction}` | Tokens per second over the exporter's own interval |
| `spark_engine_counter_reset_total{direction}` | The counter went backwards; no rate is emitted |
| `spark_engine_requests_running`, `_requests_waiting` | Native queue depth |
| `spark_engine_kv_cache_usage_ratio` | Cache occupancy as a ratio |
| `spark_engine_<stage>_seconds_mean`, `_observations_total` | Arithmetic mean and count from the native histogram's `_sum` and `_count` |
| `spark_engine_mtp_draft_tokens_total`, `_accepted_tokens_total` | Speculative decoding, where the engine counts it |
| `spark_engine_speculation_acceptance_ratio` | Accepted divided by drafted |

Rates come from the exporter's own observation interval, so a restart produces a
reset counter rather than a spike. Percentiles stay in Prometheus, where the
buckets are; `mean` is the true mean of observed requests and is never presented
as a percentile.

The exact counter families, engine prerequisites and pinned sources are in
[native inference engines](docs/inference-engines.md). The canonical recording
rules are `prometheus/engine-rules.yml`, installed as
`/usr/share/spark-exporter/prometheus/engine-rules.yml`; add that path to your
Prometheus `rule_files` to get `spark:engine_input_tokens:rate5m` and
`spark:engine_output_tokens:rate5m`. The package does not edit or reload a
Prometheus you own.

Training frameworks are documented separately in
[training telemetry](docs/training-telemetry.md). Ray Train, Kubeflow's training
operator, MLflow and Determined expose Prometheus endpoints carrying
job-lifecycle or server metrics; none of them expose per-step loss through a
passive standard endpoint, and Unsloth's Studio reports JSON under an
authenticated session. The exporter collects category-one endpoints only, and
that collector is not yet implemented.

## How readings are reported

Every scrape is a fresh observation. A collector that fails emits no values and
sets `spark_collector_success{collector=...}` to 0 for that scrape. RDMA reads
that partly fail keep their valid samples and count the failures in
`spark_rdma_read_errors`. `spark_collector_last_success_timestamp_seconds`
records the last successful observation in the current process, and 0 means none
yet.

Advertised link rates are capacity, not traffic. PCIe `GT/s` counts transfers and
ignores encoding overhead. `ethtool` statistics are exported as a gauge because
drivers mix counters and instantaneous values, so apply `rate()` only where the
driver documents a monotonic counter. A USB `speed` value is the negotiated
signalling rate, not measured throughput.

On GB10, `MemAvailable` is the kernel's estimate of allocatable memory, and
`MemTotal - MemAvailable` is a pressure signal rather than a measure of model
residency. Per-process GPU memory is driver accounting, so summing it against
system memory double-counts the same pool.

Virtual block devices — loop, device-mapper, md, zram and optical — belong to
their backing store and carry no bus of their own, so they are not reported as
physical disks. Partitions are attributes of their whole disk.

Mounts are enumerated from the service's own mount namespace, so the report
describes what the exporter can actually see. The unit sets `ProtectHome=yes`,
which replaces `/home` with an empty read-only tmpfs, so a filesystem mounted
under a home directory is absent from the report even when it is mounted on the
host. Mounts outside `/home` are visible read-only.

## Metric catalogue

Every `spark_*` family is a gauge unless marked as a counter. `node_*` families
keep upstream names and semantics.

| Family | Labels and units |
|---|---|
| `spark_host_info` | `architecture`, `version`; constant 1 |
| `spark_gpu_info` | `gpu`, `uuid`, `name`; constant 1 |
| `spark_gpu_identity` | `gpu`, `uuid`, `kind`, `detail`; constant 1 |
| `spark_gpu_field_supported` | `gpu`, `uuid`, `field`; 0 when the driver does not support the field |
| `spark_gpu_temperature_celsius` | `gpu`, `uuid`; degrees Celsius |
| `spark_gpu_temperature_limit_celsius` | `gpu`, `uuid`; driver slow-down limit |
| `spark_gpu_temperature_slow_limit_celsius` | `gpu`, `uuid` |
| `spark_gpu_temperature_max_limit_celsius` | `gpu`, `uuid` |
| `spark_gpu_temperature_target_celsius` | `gpu`, `uuid` |
| `spark_gpu_memory_temperature_celsius` | `gpu`, `uuid` |
| `spark_gpu_power_watts` | `gpu`, `uuid`; instantaneous draw |
| `spark_gpu_power_average_watts` | `gpu`, `uuid` |
| `spark_gpu_power_limit_watts` | `gpu`, `uuid`; current limit |
| `spark_gpu_power_requested_limit_watts` | `gpu`, `uuid` |
| `spark_gpu_power_default_limit_watts` | `gpu`, `uuid` |
| `spark_gpu_power_min_limit_watts` | `gpu`, `uuid` |
| `spark_gpu_power_max_limit_watts` | `gpu`, `uuid` |
| `spark_gpu_memory_power_watts` | `gpu`, `uuid` |
| `spark_gpu_module_power_watts` | `gpu`, `uuid` |
| `spark_gpu_clock_graphics_hertz` | `gpu`, `uuid` |
| `spark_gpu_clock_sm_hertz` | `gpu`, `uuid` |
| `spark_gpu_clock_memory_hertz` | `gpu`, `uuid` |
| `spark_gpu_clock_video_hertz` | `gpu`, `uuid` |
| `spark_gpu_clock_applications_graphics_hertz` | `gpu`, `uuid` |
| `spark_gpu_clock_max_graphics_hertz` | `gpu`, `uuid` |
| `spark_gpu_clock_max_sm_hertz` | `gpu`, `uuid` |
| `spark_gpu_clock_max_video_hertz` | `gpu`, `uuid` |
| `spark_gpu_memory_total_bytes` | `gpu`, `uuid`; framebuffer |
| `spark_gpu_memory_used_bytes` | `gpu`, `uuid`; framebuffer |
| `spark_gpu_memory_free_bytes` | `gpu`, `uuid`; framebuffer |
| `spark_gpu_memory_reserved_bytes` | `gpu`, `uuid`; framebuffer |
| `spark_gpu_bar1_memory_total_bytes` | `gpu`, `uuid` |
| `spark_gpu_bar1_memory_used_bytes` | `gpu`, `uuid` |
| `spark_gpu_utilization_ratio` | `gpu`, `uuid`; 0 to 1 |
| `spark_gpu_memory_utilization_ratio` | `gpu`, `uuid`; 0 to 1 |
| `spark_gpu_encoder_utilization_ratio` | `gpu`, `uuid`; 0 to 1 |
| `spark_gpu_decoder_utilization_ratio` | `gpu`, `uuid`; 0 to 1 |
| `spark_gpu_jpeg_utilization_ratio` | `gpu`, `uuid`; 0 to 1 |
| `spark_gpu_ofa_utilization_ratio` | `gpu`, `uuid`; 0 to 1 |
| `spark_gpu_pstate` | `gpu`, `uuid`; P0 is 0 |
| `spark_gpu_persistence_mode` | `gpu`, `uuid`; 0 or 1 |
| `spark_gpu_c2c_mode` | `gpu`, `uuid`; 0 or 1 |
| `spark_gpu_clock_throttle_active` | `gpu`, `uuid`, `reason`; 0 or 1 |
| `spark_gpu_clock_throttle_seconds_total` | **counter**, `gpu`, `uuid`, `reason`; seconds |
| `spark_gpu_pcie_generation` | `gpu`, `uuid`; generation number |
| `spark_gpu_pcie_max_generation` | `gpu`, `uuid` |
| `spark_gpu_pcie_device_max_generation` | `gpu`, `uuid` |
| `spark_gpu_pcie_host_max_generation` | `gpu`, `uuid` |
| `spark_gpu_pcie_link_width_lanes` | `gpu`, `uuid` |
| `spark_gpu_pcie_max_link_width_lanes` | `gpu`, `uuid` |
| `spark_gpu_pcie_replay_total` | **counter**, `gpu`, `uuid` |
| `spark_gpu_pcie_replay_rollover_total` | **counter**, `gpu`, `uuid` |
| `spark_gpu_ecc_sram_correctable_total` | **counter**, `gpu`, `uuid` |
| `spark_gpu_ecc_sram_uncorrectable_total` | **counter**, `gpu`, `uuid` |
| `spark_gpu_ecc_dram_correctable_total` | **counter**, `gpu`, `uuid` |
| `spark_gpu_ecc_dram_uncorrectable_total` | **counter**, `gpu`, `uuid` |
| `spark_gpu_process_memory_bytes` | `pid`, `uuid`; bytes |
| `spark_gpu_process_memory_supported` | `pid`, `uuid`; 1 when the driver reports bytes for this process |
| `spark_system_memory_total_bytes` | bytes |
| `spark_system_memory_available_bytes` | bytes; kernel allocatable estimate |
| `spark_system_memory_free_bytes` | bytes |
| `spark_system_memory_unavailable_bytes` | bytes; `total - available` |
| `spark_system_memory_buffers_bytes` | bytes |
| `spark_system_memory_cached_bytes` | bytes |
| `spark_system_memory_reclaimable_slab_bytes` | bytes |
| `spark_system_memory_shared_bytes` | bytes |
| `spark_system_memory_swap_total_bytes` | bytes |
| `spark_system_memory_swap_free_bytes` | bytes |
| `spark_thermal_zone_celsius` | `zone`, `type`; degrees Celsius |
| `spark_hwmon_temperature_celsius` | `chip`, `sensor`, `label`; degrees Celsius |
| `spark_hwmon_temperature_critical_celsius` | `chip`, `sensor`, `label` |
| `spark_hwmon_temperature_max_celsius` | `chip`, `sensor`, `label` |
| `spark_hwmon_temperature_min_celsius` | `chip`, `sensor`, `label` |
| `spark_thermal_read_errors` | failed reads in this collection |
| `spark_cpu_frequency_current_hertz` | `policy`, `cpu` |
| `spark_cpu_frequency_configured_max_hertz` | `policy`, `cpu` |
| `spark_cpu_frequency_configured_min_hertz` | `policy`, `cpu` |
| `spark_cpu_frequency_hardware_max_hertz` | `policy`, `cpu` |
| `spark_cpu_frequency_governor` | `policy`, `cpu`, `governor`; current governor is 1 |
| `spark_cpu_read_errors` | failed reads in this collection |
| `spark_nvme_identity` | `controller`, `kind`, `detail`; constant 1 |
| `spark_nvme_read_errors` | failed reads in this collection |
| `spark_nic_info` | `device`, plus `driver` and `bus` when readable; constant 1 |
| `spark_nic_speed_bits_per_second` | `device`; negotiated bits per second |
| `spark_nic_carrier` | `device`; 0 or 1 |
| `spark_nic_mtu_bytes` | `device`; bytes |
| `spark_nic_operstate` | `device`, `state`; the current state is 1 |
| `spark_nic_duplex` | `device`, `state`; the current state is 1 |
| `spark_nic_pcie_current_link_speed_transfers_per_second` | `device`; transfers per second, not payload bits |
| `spark_nic_pcie_max_link_speed_transfers_per_second` | `device` |
| `spark_nic_pcie_current_link_width_lanes` | `device` |
| `spark_nic_pcie_max_link_width_lanes` | `device` |
| `spark_nic_receive_errors_total` | **counter**, `device` |
| `spark_nic_transmit_errors_total` | **counter**, `device` |
| `spark_nic_receive_dropped_total` | **counter**, `device` |
| `spark_nic_transmit_dropped_total` | **counter**, `device` |
| `spark_nic_collisions_total` | **counter**, `device` |
| `spark_nic_receive_crc_errors_total` | **counter**, `device` |
| `spark_nic_receive_frame_errors_total` | **counter**, `device` |
| `spark_nic_receive_missed_errors_total` | **counter**, `device` |
| `spark_nic_transmit_carrier_errors_total` | **counter**, `device` |
| `spark_nic_receive_length_errors_total` | **counter**, `device` |
| `spark_nic_receive_over_errors_total` | **counter**, `device` |
| `spark_rdma_state` | `device`, `port`; 4 is ACTIVE |
| `spark_rdma_phys_state` | `device`, `port`; 5 is LinkUp |
| `spark_rdma_rate_bits_per_second` | `device`, `port`; advertised rate |
| `spark_rdma_counter_total` | **counter**, `device`, `port`, `source`, `counter`, `unit` |
| `spark_rdma_read_errors` | failed reads in this collection |
| `spark_ethtool_stat` | `device`, `stat`; raw driver value, gauge |
| `spark_block_device_info` | `device`, `transport`, `usb_path` when USB; constant 1 |
| `spark_block_device_size_bytes` | `device`; bytes |
| `spark_block_device_removable` | `device`; 0 or 1 |
| `spark_block_device_rotational` | `device`; 0 or 1 |
| `spark_block_device_identity` | `device`, `kind`, `detail`; constant 1 |
| `spark_block_device_read_errors` | failed reads in this collection |
| `spark_usb_device_info` | `path`, `bus`, `address`, `id_vendor`, `id_product`, plus `manufacturer`, `product`, `serial` when readable; constant 1 |
| `spark_usb_device_speed_bits_per_second` | `path`; negotiated signalling rate |
| `spark_usb_read_errors` | failed reads in this collection |
| `spark_external_filesystem_info` | `device`, `mountpoint`, `fstype`, `transport`; constant 1 |
| `spark_external_filesystem_removable` | same labels; 0 or 1 |
| `spark_external_filesystem_read_errors` | failed reads in this collection |
| `spark_supervisor_memory_budget_bytes` | admission budget; opt-in |
| `spark_supervisor_memory_reserved_bytes` | declared reservations; opt-in |
| `spark_supervisor_leases` | lease count; opt-in |
| `spark_supervisor_auto_evict` | 0 or 1; opt-in |
| `spark_supervisor_ready_alias_count` | opt-in |
| `spark_supervisor_lane_ready` | `lane`, `model`; 0 or 1 |
| `spark_supervisor_lane_state` | `lane`, `model`, `state`; the current state is 1 |
| `spark_supervisor_lane_reserved` | `lane`, `model` |
| `spark_supervisor_lane_draining` | `lane`, `model` |
| `spark_supervisor_lane_leases` | `lane`, `model` |
| `spark_supervisor_lane_memory_declared_bytes` | `lane`, `model`; declared, not measured |
| `spark_supervisor_lane_unit_found` | `lane`, `model` |
| `spark_supervisor_lane_unit_check_error` | `lane`, `model` |
| `spark_supervisor_lane_port` | `lane`, `model` |
| `spark_supervisor_lane_priority` | `lane`, `model` |
| `spark_supervisor_lane_elapsed_seconds` | `lane`, `model` |
| `spark_supervisor_lane_fence_count` | `lane`, `model` |
| `spark_supervisor_lane_identity` | `lane`, `model`, `kind`, `detail`; constant 1 |
| `spark_engine_tokens_total` | **counter**, `engine`, `engine_port`, `model`, `direction` |
| `spark_engine_tokens_per_second` | `engine`, `engine_port`, `model`, `direction` |
| `spark_engine_counter_reset_total` | **counter**, `engine`, `engine_port`, `model`, `direction` |
| `spark_engine_endpoint_up` | `engine`, `engine_port`, `model`; constant 1 |
| `spark_engine_requests_running` | `engine`, `engine_port`, `model` |
| `spark_engine_requests_waiting` | `engine`, `engine_port`, `model` |
| `spark_engine_generation_tokens_running` | `engine`, `engine_port`, `model` |
| `spark_engine_kv_cache_usage_ratio` | `engine`, `engine_port`, `model`; 0 to 1 |
| `spark_engine_time_to_first_token_seconds_mean` | `engine`, `engine_port`, `model`; seconds |
| `spark_engine_time_to_first_token_seconds_observations_total` | `engine`, `engine_port`, `model` |
| `spark_engine_inter_token_seconds_mean` | `engine`, `engine_port`, `model`; seconds |
| `spark_engine_inter_token_seconds_observations_total` | `engine`, `engine_port`, `model` |
| `spark_engine_prefill_seconds_mean` | `engine`, `engine_port`, `model`; seconds |
| `spark_engine_prefill_seconds_observations_total` | `engine`, `engine_port`, `model` |
| `spark_engine_e2e_request_seconds_mean` | `engine`, `engine_port`, `model`; seconds |
| `spark_engine_e2e_request_seconds_observations_total` | `engine`, `engine_port`, `model` |
| `spark_engine_mtp_draft_tokens_total` | **counter**, `engine`, `engine_port`, `model` |
| `spark_engine_mtp_accepted_tokens_total` | **counter**, `engine`, `engine_port`, `model` |
| `spark_engine_spec_draft_tokens_total` | **counter**, `engine`, `engine_port`, `model` |
| `spark_engine_spec_accepted_tokens_total` | **counter**, `engine`, `engine_port`, `model` |
| `spark_engine_speculation_acceptance_ratio` | `engine`, `engine_port`, `model`; 0 to 1 |
| `spark_collector_success` | `collector`; 0 or 1 |
| `spark_collector_duration_seconds` | `collector` |
| `spark_collector_last_success_timestamp_seconds` | `collector`; Unix seconds, 0 means none yet |

Collector names are `gpu`, `gpu_processes`, `memory`, `thermal`, `cpu`, `nvme`,
`nics`, `rdma`, `ethtool`, `block_devices`, `usb`, `external_filesystems`, and
`supervisor`, `engine_stats`, `node_exporter` when those are enabled.

## Build

```sh
sudo apt install build-essential dpkg-dev debhelper dh-python python3-all
make vendor DEB_HOST_ARCH=arm64    # or amd64
dpkg-buildpackage --build=binary --no-sign
```

`make vendor` downloads the pinned node_exporter 1.12.1 release, checks it
against the committed SHA-256, and stages the binary with its license notices.
The build itself runs offline and the package ships those bytes untouched.
`NODE_EXPORTER_ARCHIVE=/path/to/tarball` prepares from a local archive.

The artifact lands in the parent directory as
`spark-exporter_0.1.3-1_arm64.deb`. The bundled binary is architecture-specific,
so build on the target architecture or in a configured cross-build environment.

## Tests

```sh
python3 -B -m unittest discover -v
```

The suite covers missing-data semantics, RDMA units and partial failure, NIC
metadata without RDMA, raw driver statistics, PCIe fields, block-device and USB
attribution, external mount detection, engine rate and reset behaviour, bind
address validation, resource closure, byte-preserved histograms, private node
merging, proxy isolation and single-flight discovery. `--root /path/to/fixture`
points the collectors at a fixture `proc/` and `sys/` tree, and the command and
HTTP callables are injectable.

Continuous integration runs the suite on x86-64 and arm64 across supported Python
versions, then builds the Debian package, installs it, scrapes the live endpoint
and purges it to confirm cleanup. Tagging a release builds both architectures,
attests provenance, and publishes the packages with a `SHA256SUMS` file.

## References

- [NVIDIA DGX Spark and GB10 specifications](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
- [NVIDIA System Management Interface](https://docs.nvidia.com/deploy/nvidia-smi/index.html)
- [Linux meminfo](https://docs.kernel.org/filesystems/proc.html)
- [Linux network interface sysfs ABI](https://www.kernel.org/doc/Documentation/ABI/testing/sysfs-class-net)
- [Linux PCI sysfs ABI](https://www.kernel.org/doc/Documentation/ABI/testing/sysfs-bus-pci)
- [Linux block device sysfs ABI](https://www.kernel.org/doc/Documentation/ABI/testing/sysfs-block)
- [Linux USB sysfs ABI](https://www.kernel.org/doc/Documentation/ABI/stable/sysfs-bus-usb)
- [Linux thermal sysfs ABI](https://www.kernel.org/doc/Documentation/ABI/testing/sysfs-class-thermal)
- [Linux network statistics and ethtool](https://docs.kernel.org/networking/statistics.html)
- [Linux InfiniBand sysfs ABI](https://www.kernel.org/doc/Documentation/ABI/stable/sysfs-class-infiniband)
- [Prometheus node_exporter](https://github.com/prometheus/node_exporter)
- [Prometheus exposition formats](https://prometheus.io/docs/instrumenting/exposition_formats/)
- [Prometheus HTTP service discovery](https://prometheus.io/docs/prometheus/latest/http_sd/)
- [Debian debhelper](https://manpages.debian.org/debhelper/debhelper.7.en.html)

License: [MIT](LICENSE). The bundled node_exporter keeps its own Apache-2.0
license and notices.
