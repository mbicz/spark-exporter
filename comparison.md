# Comparison with ateska/dgx-spark-prometheus

Compared 2026-10-26 against upstream commit
[`b59abc5eeadf0edcf956cb56bc7cce80e1745e1b`](https://github.com/ateska/dgx-spark-prometheus/tree/b59abc5eeadf0edcf956cb56bc7cce80e1745e1b).
This is a source and controlled-runtime comparison, not a performance benchmark
and not a claim about readings on physical Spark hardware.

## Why this exporter exists

[ateska/dgx-spark-prometheus](https://github.com/ateska/dgx-spark-prometheus) is
a standalone Go exporter with a useful hardware subset. It covers host CPU
usage, one thermal reading, disk operation counters and the first GPU's
utilization, temperature, clock and power. Spark Exporter covers more of what
the GB10 driver, the kernel and the inference engines actually expose, and it
reports a value only when a source produced one.

## Coverage

| Area | ateska/dgx-spark-prometheus at the compared commit | Spark Exporter |
|---|---|---|
| Implementation | Go binary with the Prometheus client library, BSD-3-Clause | Python standard-library adapter plus a pinned official node_exporter binary; systemd; no pip, container or dashboard dependency |
| Host CPU | Whole-host usage gauge, one CPU/SoC thermal reading, average clock | node_exporter per-core CPU counters, cpufreq, load, pressure, thermal and hwmon, plus per-policy frequency and governor |
| Memory | Total and used (`total - free - buffers - cached`) | Raw node memory and vmstat, plus an explicit `MemAvailable`-based unified-memory pressure series |
| Storage | Read/write operation counters and root capacity percentage | node_exporter diskstats and filesystem capacity/inodes, plus physical device transport, rotation, removable flag and identity, and USB-attached mount attribution |
| GPU | First GPU's utilization, temperature, graphics clock and power | Per-GPU UUID-labelled supported readings, every clock domain, power limits and averages, PCIe generation and width, replay and ECC counters, throttle state and accumulated durations, process memory, and a per-field support flag |
| Unsupported GPU values | Empty, N/A and parse errors become numeric zero | Unsupported field absent; `spark_gpu_field_supported` distinguishes a gap from a measured zero |
| Interface selection | Six fixed interface names, only operstate `up` | Dynamic physical-interface discovery, with kernel error, drop, collision and CRC counters |
| 200 Gb fabric | RX/TX bytes and packets for those interfaces | Negotiated link capacity, state, MTU and PCIe width, raw RDMA counters and per-driver ethtool statistics including PFC, CNP, ECN and error counters |
| Inference telemetry | None | Passive engine-native Prometheus/OpenMetrics text, preserving counters and histograms, with supervisor lane accounting opt-in |
| Throughput | None | Prometheus rates and recording rules over verified native counters |
| Discovery | Static targets in a README example | Prometheus HTTP service discovery served by the exporter, plus bounded loopback engine probing |
| Default boundary | Service runs `-listen :9835` | Adapter defaults to loopback 9400; the packaged node_exporter is private on loopback 19100; tailnet binding is explicit opt-in |
| Privileges | Supplied unit runs as root | Unprivileged `DynamicUser` services, no model-control or container-socket access |
| Installation | `go build`, manual binary and unit copy | Debian package with declared dependencies and normal conffile and service lifecycle |

"Comprehensive" here means host, accelerator, fabric, storage and serving
telemetry that the kernel, driver and engine actually expose. SM occupancy, HBM
bandwidth, per-kernel profiling, NVMe SMART data behind privileged device access
and tokens/second on an engine without passive counters are measurement
boundaries, not zero values. Native engine metrics may require an engine flag;
installing this package does not reconfigure or restart inference.

## Controlled parser run

The unmodified upstream Go exporter was built and started with
`-listen 127.0.0.1:19835`. A throwaway `nvidia-smi` executable emitted the
fixture `N/A, 52, 61.5, 1200`. Scraping the running `/metrics` endpoint returned
`gpu_utilization_percent{host="mike.internal"} 0` while temperature, power and
clock matched the fixture fields. The N/A became a measured-looking zero. This
demonstrates the parsing behaviour; it is not an actual GPU measurement.

Source inspection also shows `exec.Command(...).Output()` without a context or
timeout, and a supplied service that runs as root on a wildcard port. The Python
adapter bounds its subprocess calls and issues only read-only local requests.

## Sources

- [GPU collection and N/A parsing](https://github.com/ateska/dgx-spark-prometheus/blob/b59abc5eeadf0edcf956cb56bc7cce80e1745e1b/collectors/gpu.go)
- [Fixed network interface inventory](https://github.com/ateska/dgx-spark-prometheus/blob/b59abc5eeadf0edcf956cb56bc7cce80e1745e1b/collectors/network.go)
- [Registered collectors and listener default](https://github.com/ateska/dgx-spark-prometheus/blob/b59abc5eeadf0edcf956cb56bc7cce80e1745e1b/main.go)
- [Service privileges and binding](https://github.com/ateska/dgx-spark-prometheus/blob/b59abc5eeadf0edcf956cb56bc7cce80e1745e1b/dgx-spark-prometheus.service)
- [CPU](https://github.com/ateska/dgx-spark-prometheus/blob/b59abc5eeadf0edcf956cb56bc7cce80e1745e1b/collectors/cpu.go), [memory](https://github.com/ateska/dgx-spark-prometheus/blob/b59abc5eeadf0edcf956cb56bc7cce80e1745e1b/collectors/memory.go), [disk](https://github.com/ateska/dgx-spark-prometheus/blob/b59abc5eeadf0edcf956cb56bc7cce80e1745e1b/collectors/disk.go)
- [node_exporter collectors](https://github.com/prometheus/node_exporter#collectors)
- [NVIDIA SMI reference](https://docs.nvidia.com/deploy/nvidia-smi/)
- [Linux InfiniBand sysfs ABI](https://www.kernel.org/doc/Documentation/ABI/stable/sysfs-class-infiniband)
