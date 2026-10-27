#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Read-only Spark hardware exporter and bounded loopback inference metrics proxy."""
import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import csv
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import threading
import time
import xml.etree.ElementTree as ET

VERSION = "0.1.0"
MAX_RESPONSE = 32 * 1024 * 1024
COMMON_ENGINE_PORTS = frozenset((8080, 9931, 30000))
ENGINE_METRICS_PATHS = ("/prometheus/metrics", "/metrics")
PROM_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
NUMBER = r"(?:[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?Inf|NaN)"
LABEL = r'[a-zA-Z_][a-zA-Z0-9_]*="(?:[^"\\\n]|\\[\\"n])*"'
LABELS = r"\s*(?:" + LABEL + r"(?:\s*,\s*" + LABEL + r")*\s*,?)?\s*"
SAMPLE = re.compile(r"[a-zA-Z_:][a-zA-Z0-9_:]*(?:\{" + LABELS + r"\})?\s+" + NUMBER + r"(?:\s+" + NUMBER + r")?(?:\s+#\s+\{" + LABELS + r"\}\s+" + NUMBER + r"(?:\s+" + NUMBER + r")?)?\s*$")
TAILNET = ipaddress.ip_network("100.64.0.0/10")


def listen_ipv4(value, command=None):
    if value == "tailscale":
        value = (command or run_command)(["tailscale", "ip", "-4"]).strip()
        address = ipaddress.ip_address(value)
        if address.version != 4 or address not in TAILNET:
            raise ValueError("tailscale must report one IPv4 in 100.64.0.0/10")
    else:
        address = ipaddress.ip_address(value)
    if address.version != 4 or address.is_unspecified or address.is_multicast or str(address) == "255.255.255.255":
        raise ValueError("a concrete unicast IPv4 listen address is required")
    return str(address)


def run_command(args):
    # No shell, inherited credentials, command lines, or stderr in responses/logs.
    result = subprocess.run(args, capture_output=True, text=True, timeout=4,
                            env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "LC_ALL": "C"}, check=True)
    return result.stdout


def number(value):
    if value is None:
        return None
    match = re.fullmatch(r"\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))(?:\s+[^\d].*)?\s*", value)
    return float(match[1]) if match else None


def unescape_mount(value):
    """Decode the octal escapes the kernel uses for spaces and tabs in mount paths."""
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)


def sample(metric, value, **labels):
    if isinstance(value, bool):
        value = int(value)
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("invalid numeric metric")
    suffix = ""
    if labels:
        def quote(value):
            return '"' + str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"') + '"'
        suffix = "{" + ",".join(key + "=" + quote(val) for key, val in sorted(labels.items())) + "}"
    return f"{metric}{suffix} {value}\n"


class HTTPStatusError(ValueError):
    """An upstream status without retaining private response bodies or headers."""
    def __init__(self, status):
        super().__init__("upstream HTTP request did not succeed")
        self.status = status


def fetch_local(port, path, limit=MAX_RESPONSE):
    # HTTPConnection bypasses proxy env, redirects and incoming Authorization headers.
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1.5)
    try:
        accept = "application/json" if path == "/openapi.json" else "application/openmetrics-text, text/plain"
        connection.request("GET", path, headers={"Accept": accept, "Accept-Encoding": "identity"})
        with connection.getresponse() as response:
            if response.status != 200:
                raise HTTPStatusError(response.status)
            body = response.read(limit + 1)
            if len(body) > limit:
                raise ValueError("upstream response exceeds limit")
            return body, response.getheader("Content-Type", "")
    finally:
        connection.close()


def validate_metrics(body, content_type):
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type not in ("text/plain", "application/openmetrics-text"):
        raise ValueError("not a Prometheus text response")
    lines = body.decode("utf-8").splitlines()
    count = 0
    for line in lines:
        if not line or line.startswith("#"):
            continue
        if not SAMPLE.fullmatch(line):
            raise ValueError("invalid Prometheus sample")
        count += 1
    if not count:
        raise ValueError("empty metrics endpoint")
    return body, content_type


SAMPLE_PARTS = re.compile(r"([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(" + LABELS + r")\})?\s+(" + NUMBER + r")")
LABEL_PARTS = re.compile(LABEL)


def parse_metrics(body):
    """Return (name, labels, value) tuples from Prometheus text, verbatim."""
    samples = []
    for line in body.decode("utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        match = SAMPLE_PARTS.match(line)
        if not match:
            continue
        labels = {}
        for label in LABEL_PARTS.finditer(match[2] or ""):
            key, _, raw = label[0].partition("=")
            labels[key] = raw[1:-1].replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
        try:
            samples.append((match[1], labels, float(match[3])))
        except ValueError:
            continue
    return samples


# Exact native cumulative token counters per engine, with the label carrying the
# native model identity. Selection matches prometheus/engine-rules.yml so the
# exporter's own view and the Prometheus records agree.
ENGINE_TOKEN_COUNTERS = (
    ("vllm", "vllm:prompt_tokens_total", "vllm:generation_tokens_total", "model_name", {}),
    ("sglang", "sglang:prompt_tokens_total", "sglang:generation_tokens_total", "model_name", {}),
    ("llamacpp", "llamacpp:prompt_tokens_total", "llamacpp:tokens_predicted_total", None, {}),
    ("tensorrt-llm", "trtllm_prompt_tokens_total", "trtllm_generation_tokens_total", "model_name", {}),
    ("triton-tensorrt-llm", "nv_llm_input_token_len_sum", "nv_llm_output_token_len_sum", "model",
     {"response_metric_type": "total_input_tokens"}),
    ("tensorfold", "tensorfold:prompt_tokens_total", "tensorfold:generation_tokens_total", None, {}),
)

# Native gauge families, already carrying the model label where the engine has one.
ENGINE_GAUGES = (
    ("vllm", "vllm:num_requests_running", "requests_running"),
    ("vllm", "vllm:num_requests_waiting", "requests_waiting"),
    ("sglang", "sglang:num_running_reqs", "requests_running"),
    ("sglang", "sglang:num_queue_reqs", "requests_waiting"),
    ("llamacpp", "llamacpp:requests_processing", "requests_running"),
    ("llamacpp", "llamacpp:requests_deferred", "requests_waiting"),
    ("tensorrt-llm", "trtllm_num_requests_running", "requests_running"),
    ("tensorrt-llm", "trtllm_num_requests_waiting", "requests_waiting"),
    ("tensorfold", "tensorfold:requests_running", "requests_running"),
    ("tensorfold", "tensorfold:requests_waiting", "requests_waiting"),
    ("tensorfold", "tensorfold:generation_tokens_running", "generation_tokens_running"),
)

# KV-cache occupancy is a ratio on current engines, a percentage on older vLLM.
ENGINE_CACHE = (
    ("vllm", "vllm:kv_cache_usage_perc", "kv_cache_usage_ratio", 1),
    ("vllm", "vllm:gpu_cache_usage_perc", "kv_cache_usage_ratio", 1),
    ("sglang", "sglang:token_usage", "kv_cache_usage_ratio", 1),
    ("tensorfold", "tensorfold:kv_cache_usage_ratio", "kv_cache_usage_ratio", 1),
    ("tensorfold", "tensorfold:kv_cache_usage_perc", "kv_cache_usage_ratio", 0.01),
)

# Histogram families whose _sum/_count pair yields an honest arithmetic mean.
# Percentiles stay in Prometheus, which needs the buckets.
ENGINE_LATENCY = (
    ("vllm", "vllm:time_to_first_token_seconds", "time_to_first_token_seconds"),
    ("vllm", "vllm:inter_token_latency_seconds", "inter_token_seconds"),
    ("vllm", "vllm:e2e_request_latency_seconds", "e2e_request_seconds"),
    ("sglang", "sglang:time_to_first_token_seconds", "time_to_first_token_seconds"),
    ("sglang", "sglang:e2e_request_latency_seconds", "e2e_request_seconds"),
    ("tensorfold", "tensorfold:time_to_first_token_seconds", "time_to_first_token_seconds"),
    ("tensorfold", "tensorfold:request_time_per_output_token_seconds", "inter_token_seconds"),
    ("tensorfold", "tensorfold:e2e_request_latency_seconds", "e2e_request_seconds"),
    ("tensorfold", "tensorfold:request_prefill_time_seconds", "prefill_seconds"),
)

# Speculative-decoding acceptance is meaningful only where the engine exposes it.
ENGINE_SPECULATION = (
    ("tensorfold", "tensorfold:mtp_drafted_total", "mtp_draft_tokens_total"),
    ("tensorfold", "tensorfold:mtp_accepted_total", "mtp_accepted_tokens_total"),
    ("tensorfold", "tensorfold:spec_decode_num_draft_tokens_total", "spec_draft_tokens_total"),
    ("tensorfold", "tensorfold:spec_decode_num_accepted_tokens_total", "spec_accepted_tokens_total"),
)



class SingleFlight:
    """Share overlapping requests only; the next request always probes again."""
    def __init__(self):
        self.lock = threading.Lock()
        self.pending = {}

    def call(self, key, function):
        with self.lock:
            future = self.pending.get(key)
            owner = future is None
            if owner:
                future = self.pending[key] = Future()
        if owner:
            try:
                future.set_result(function())
            except Exception as error:
                future.set_exception(error)
            finally:
                with self.lock:
                    del self.pending[key]
        return future.result()


class Exporter:
    def __init__(self, address, root="/", config_dir="/etc/spark-exporter/lanes.d",
                 worker_range=(8000, 8031), command=run_command, fetch=fetch_local,
                 *, enable_supervisor=False, enable_inference=False, node_exporter_url=None,
                 engine_endpoints=()):
        self.address = listen_ipv4(address, command)
        self.enable_supervisor = enable_supervisor
        self.enable_inference = enable_inference
        self.root = Path(root)
        self.config_dir = Path(config_dir)
        low, high = worker_range
        if not 1 <= low <= high <= 65535 or high - low >= 128:
            raise ValueError("worker range must contain 1..128 valid ports")
        self.worker_range = (low, high)
        self.command = command
        self.fetch = fetch
        self.engine_endpoints = {}
        for endpoint in engine_endpoints:
            match = re.fullmatch(r"([1-9][0-9]{0,4}):(/metrics|/prometheus/metrics)", endpoint)
            if not match or int(match[1]) > 65535:
                raise ValueError("engine endpoint must be PORT:/metrics or PORT:/prometheus/metrics")
            port = int(match[1])
            if port in self.engine_endpoints:
                raise ValueError("engine endpoint port is configured more than once")
            self.engine_endpoints[port] = match[2]
        self.node_exporter_port = None
        if node_exporter_url is not None:
            match = re.fullmatch(r"http://127\.0\.0\.1:([1-9][0-9]{0,4})/metrics", node_exporter_url)
            if not match or not 1 <= int(match[1]) <= 65535 or int(match[1]) == 9400:
                raise ValueError("node exporter URL must be http://127.0.0.1:<port>/metrics (not port 9400)")
            self.node_exporter_port = int(match[1])
        if self.engine_endpoints.keys() & {8090, 9100, 9400, self.node_exporter_port}:
            raise ValueError("engine endpoint uses a reserved collector port")
        self.flights = SingleFlight()
        self.health_lock = threading.Lock()
        self.last_success = {}
        self.engine_lock = threading.Lock()
        self.engine_counters = {}

    def path(self, name):
        return self.root / name.lstrip("/")

    def gpu(self):
        document = ET.fromstring(self.command(["nvidia-smi", "-q", "-x"]))
        result = []
        fields = {
            "temperature_celsius": (["temperature/gpu_temp"], 1),
            "temperature_limit_celsius": (["temperature/gpu_temp_tlimit"], 1),
            "temperature_slow_limit_celsius": (["temperature/gpu_temp_slow_tlimit_threshold"], 1),
            "temperature_max_limit_celsius": (["temperature/gpu_temp_max_tlimit_threshold"], 1),
            "temperature_target_celsius": (["temperature/gpu_target_temperature"], 1),
            "memory_temperature_celsius": (["temperature/memory_temp"], 1),
            "power_watts": (["gpu_power_readings/instant_power_draw", "gpu_power_readings/power_draw", "power_readings/power_draw"], 1),
            "power_average_watts": (["gpu_power_readings/average_power_draw"], 1),
            "power_limit_watts": (["gpu_power_readings/current_power_limit", "gpu_power_readings/power_limit", "power_readings/power_limit"], 1),
            "power_requested_limit_watts": (["gpu_power_readings/requested_power_limit"], 1),
            "power_default_limit_watts": (["gpu_power_readings/default_power_limit"], 1),
            "power_min_limit_watts": (["gpu_power_readings/min_power_limit"], 1),
            "power_max_limit_watts": (["gpu_power_readings/max_power_limit"], 1),
            "memory_power_watts": (["gpu_memory_power_readings/instant_power_draw"], 1),
            "module_power_watts": (["module_power_readings/instant_power_draw"], 1),
            "clock_graphics_hertz": (["clocks/graphics_clock"], 1e6),
            "clock_sm_hertz": (["clocks/sm_clock"], 1e6),
            "clock_memory_hertz": (["clocks/mem_clock"], 1e6),
            "clock_video_hertz": (["clocks/video_clock"], 1e6),
            "clock_applications_graphics_hertz": (["applications_clocks/graphics_clock"], 1e6),
            "clock_max_graphics_hertz": (["max_clocks/graphics_clock"], 1e6),
            "clock_max_sm_hertz": (["max_clocks/sm_clock"], 1e6),
            "clock_max_video_hertz": (["max_clocks/video_clock"], 1e6),
            "memory_total_bytes": (["fb_memory_usage/total"], 1024 ** 2),
            "memory_used_bytes": (["fb_memory_usage/used"], 1024 ** 2),
            "memory_free_bytes": (["fb_memory_usage/free"], 1024 ** 2),
            "memory_reserved_bytes": (["fb_memory_usage/reserved"], 1024 ** 2),
            "bar1_memory_total_bytes": (["bar1_memory_usage/total"], 1024 ** 2),
            "bar1_memory_used_bytes": (["bar1_memory_usage/used"], 1024 ** 2),
            "utilization_ratio": (["utilization/gpu_util"], .01),
            "memory_utilization_ratio": (["utilization/memory_util"], .01),
            "encoder_utilization_ratio": (["utilization/encoder_util"], .01),
            "decoder_utilization_ratio": (["utilization/decoder_util"], .01),
            "jpeg_utilization_ratio": (["utilization/jpeg_util"], .01),
            "ofa_utilization_ratio": (["utilization/ofa_util"], .01),
            "pcie_generation": (["pci/pci_gpu_link_info/pcie_gen/current_link_gen"], 1),
            "pcie_max_generation": (["pci/pci_gpu_link_info/pcie_gen/max_link_gen"], 1),
            "pcie_device_max_generation": (["pci/pci_gpu_link_info/pcie_gen/max_device_link_gen"], 1),
            "pcie_host_max_generation": (["pci/pci_gpu_link_info/pcie_gen/max_host_link_gen"], 1),
            "pcie_link_width_lanes": (["pci/pci_gpu_link_info/link_widths/current_link_width"], 1),
            "pcie_max_link_width_lanes": (["pci/pci_gpu_link_info/link_widths/max_link_width"], 1),
            "pcie_replay_total": (["pci/replay_counter"], 1),
            "pcie_replay_rollover_total": (["pci/replay_rollover_counter"], 1),
            "ecc_sram_correctable_total": (["ecc_errors/aggregate/sram_correctable"], 1),
            "ecc_sram_uncorrectable_total": (["ecc_errors/aggregate/sram_uncorrectable_parity"], 1),
            "ecc_dram_correctable_total": (["ecc_errors/aggregate/dram_correctable"], 1),
            "ecc_dram_uncorrectable_total": (["ecc_errors/aggregate/dram_uncorrectable"], 1),
        }
        devices = document.findall("gpu")
        if not devices:
            raise ValueError("no GPUs reported")
        for index, gpu in enumerate(devices):
            labels = {"gpu": str(index), "uuid": gpu.findtext("uuid", "unknown")}
            result.append(sample("spark_gpu_info", 1, name=gpu.findtext("product_name", "unknown"), **labels))
            for field, (paths, scale) in fields.items():
                values = (gpu.findtext(path) for path in paths)
                if field.endswith("width_lanes"):
                    values = (re.sub(r"(?<=\d)x$", "", raw.strip()) if raw else None for raw in values)
                value = next((value for raw in values if (value := number(raw)) is not None), None)
                result.append(sample("spark_gpu_field_supported", value is not None, field=field, **labels))
                if value is not None:
                    result.append(sample("spark_gpu_" + field, value * scale, **labels))
            pstate = re.fullmatch(r"P(\d+)", gpu.findtext("performance_state", ""))
            result.append(sample("spark_gpu_field_supported", bool(pstate), field="pstate", **labels))
            if pstate:
                result.append(sample("spark_gpu_pstate", int(pstate[1]), **labels))
            reasons = gpu.find("clocks_event_reasons")
            if reasons is None:
                reasons = gpu.find("clocks_throttle_reasons")
            supported_reasons = 0
            if reasons is not None:
                for reason in reasons:
                    state = (reason.text or "").strip()
                    if state in ("Active", "Not Active"):
                        supported_reasons += 1
                        result.append(sample("spark_gpu_clock_throttle_active", state == "Active", reason=reason.tag, **labels))
            result.append(sample("spark_gpu_field_supported", bool(supported_reasons), field="clock_throttle", **labels))
            counters = gpu.find("clocks_event_reasons_counters")
            if counters is None:
                counters = gpu.find("clocks_throttle_reasons_counters")
            supported_counters = 0
            if counters is not None:
                for reason in counters:
                    raw = (reason.text or "").strip()
                    match = re.fullmatch(r"([0-9]+)\s*us", raw)
                    if match:
                        supported_counters += 1
                        # Driver reports accumulated microseconds in throttled state.
                        result.append(sample("spark_gpu_clock_throttle_seconds_total",
                                             int(match[1]) / 1e6, reason=reason.tag, **labels))
            result.append(sample("spark_gpu_field_supported", bool(supported_counters),
                                 field="clock_throttle_counters", **labels))
            # Non-metric identity for correlating with driver and fleet inventory.
            for element, label in (("driver_version", "driver"), ("cuda_version", "cuda"),
                                   ("vbios_version", "vbios"), ("gpu_part_number", "part"),
                                   ("board_id", "board"), ("gsp_firmware_version", "gsp")):
                value = gpu.findtext(element)
                if value and value.strip() not in ("N/A", ""):
                    result.append(sample("spark_gpu_identity", 1, kind=label, detail=value.strip(), **labels))
            persistence = (gpu.findtext("persistence_mode") or "").strip()
            if persistence in ("Enabled", "Disabled"):
                result.append(sample("spark_gpu_persistence_mode", persistence == "Enabled", **labels))
            c2c = (gpu.findtext("c2c_mode") or "").strip()
            if c2c in ("Enabled", "Disabled"):
                result.append(sample("spark_gpu_c2c_mode", c2c == "Enabled", **labels))
        return "".join(result)

    def gpu_processes(self):
        body = self.command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"])
        result = []
        for row in csv.reader(body.splitlines()):
            if not row:
                continue
            if len(row) != 3 or not row[1].strip().isdigit():
                raise ValueError("invalid compute process row")
            uuid, pid, memory = (value.strip() for value in row)
            value = number(memory)
            labels = {"uuid": uuid, "pid": pid}
            result.append(sample("spark_gpu_process_memory_supported", value is not None, **labels))
            if value is not None:
                result.append(sample("spark_gpu_process_memory_bytes", value * 1024 ** 2, **labels))
        return "".join(result)

    def memory(self):
        values = {}
        for line in self.path("/proc/meminfo").read_text().splitlines():
            key, _, raw = line.partition(":")
            if raw.strip().endswith("kB"):
                values[key] = int(raw.split()[0]) * 1024
        result = []
        for key, metric in {"MemTotal": "total", "MemAvailable": "available", "MemFree": "free",
                            "Buffers": "buffers", "Cached": "cached", "SReclaimable": "reclaimable_slab",
                            "Shmem": "shared", "SwapTotal": "swap_total", "SwapFree": "swap_free"}.items():
            if key in values:
                result.append(sample("spark_system_memory_" + metric + "_bytes", values[key]))
        if "MemTotal" in values and "MemAvailable" in values:
            result.append(sample("spark_system_memory_unavailable_bytes", values["MemTotal"] - values["MemAvailable"]))
        if "MemTotal" not in values:
            raise ValueError("no system memory")
        return "".join(result)

    def thermal(self):
        """Per-zone thermal and per-chip hwmon temperatures with labels."""
        result, failures = [], 0
        base = self.path("/sys/class/thermal")
        if base.is_dir():
            for zone in sorted(base.glob("thermal_zone*")):
                labels = {"zone": zone.name.removeprefix("thermal_zone")}
                try:
                    labels["type"] = (zone / "type").read_text().strip()
                except OSError:
                    failures += 1
                try:
                    raw = (zone / "temp").read_text().strip()
                    value = number(raw)
                    if value is None:
                        raise ValueError("no thermal reading")
                    result.append(sample("spark_thermal_zone_celsius", value / 1000.0, **labels))
                except (OSError, ValueError):
                    failures += 1
        hwmon = self.path("/sys/class/hwmon")
        if hwmon.is_dir():
            for chip in sorted(hwmon.glob("hwmon*")):
                try:
                    chip_name = (chip / "name").read_text().strip()
                except OSError:
                    failures += 1
                    continue
                # PCI-backed chips use their physical address so identical
                # driver names (four mlx5 NICs) stay distinct. Everything else
                # uses the kernel's hwmon instance name.
                resolved = chip.resolve().name
                if re.fullmatch(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", resolved):
                    chip_label = chip_name + "@" + resolved
                else:
                    chip_label = chip.name + ":" + chip_name
                for sensor in sorted(chip.glob("temp*_input")):
                    sensor_name = sensor.name.removesuffix("_input")
                    labels = {"chip": chip_label, "sensor": sensor_name}
                    try:
                        label_file = chip / (sensor_name + "_label")
                        if label_file.is_file():
                            labels["label"] = label_file.read_text().strip()
                    except OSError:
                        pass
                    try:
                        value = number(sensor.read_text().strip())
                        if value is None:
                            raise ValueError("no hwmon reading")
                        result.append(sample("spark_hwmon_temperature_celsius", value / 1000.0, **labels))
                    except (OSError, ValueError):
                        failures += 1
                    for attribute, metric in (("crit", "critical"), ("max", "max"), ("min", "min")):
                        try:
                            limit = number((chip / (sensor_name + "_" + attribute)).read_text().strip())
                            if limit is not None:
                                result.append(sample("spark_hwmon_temperature_" + metric + "_celsius",
                                                     limit / 1000.0, **labels))
                        except (OSError, ValueError):
                            pass
        result.append(sample("spark_thermal_read_errors", failures))
        if not result or len(result) == 1:
            raise FileNotFoundError("no thermal sensors")
        return "".join(result), failures == 0

    def cpu(self):
        """Per-policy frequency, governor and the configured ceiling."""
        base = self.path("/sys/devices/system/cpu/cpufreq")
        if not base.is_dir():
            raise FileNotFoundError("cpufreq unavailable")
        result, failures = [], 0
        for policy in sorted(base.glob("policy*")):
            labels = {"policy": policy.name.removeprefix("policy")}
            try:
                labels["cpu"] = (policy / "affected_cpus").read_text().strip().replace(" ", "_")
            except OSError:
                pass
            try:
                governor = (policy / "scaling_governor").read_text().strip()
                result.append(sample("spark_cpu_frequency_governor", 1, governor=governor, **labels))
            except OSError:
                failures += 1
            for field, metric in (("scaling_cur_freq", "current_hertz"),
                                  ("scaling_max_freq", "configured_max_hertz"),
                                  ("cpuinfo_max_freq", "hardware_max_hertz")):
                try:
                    value = number((policy / field).read_text().strip())
                    if value is None:
                        raise ValueError("no frequency")
                    result.append(sample("spark_cpu_frequency_" + metric, value * 1000.0, **labels))
                except (OSError, ValueError):
                    failures += 1
            # Not every driver exposes a configurable minimum.
            try:
                value = number((policy / "scaling_min_freq").read_text().strip())
                if value is not None:
                    result.append(sample("spark_cpu_frequency_configured_min_hertz", value * 1000.0, **labels))
            except (OSError, ValueError):
                pass
        result.append(sample("spark_cpu_read_errors", failures))
        return "".join(result), failures == 0

    def nvme(self):
        """NVMe identity plus the kernel-reported temperature sensors."""
        base = self.path("/sys/class/nvme")
        if not base.is_dir():
            raise FileNotFoundError("NVMe sysfs unavailable")
        result, failures = [], 0
        for controller in sorted(base.glob("nvme*")):
            labels = {"controller": controller.name}
            for field, kind in (("model", "model"), ("firmware_rev", "firmware"), ("serial", "serial")):
                source = controller / field
                # A kernel build may not expose every identity attribute; absence
                # is not a read failure.
                if not source.is_file():
                    continue
                try:
                    value = source.read_text().strip()
                    if value:
                        result.append(sample("spark_nvme_identity", 1, kind=kind, detail=value, **labels))
                except OSError:
                    failures += 1
        result.append(sample("spark_nvme_read_errors", failures))
        return "".join(result), failures == 0

    def _usb_ancestor(self, resolved):
        """The USB device a sysfs node sits behind, or None for a non-USB bus."""
        current = resolved
        while current != current.parent:
            if (current / "idVendor").is_file():
                return current
            current = current.parent
        return None

    def _transport(self, resolved):
        """Bus name from the resolved sysfs path; USB is a walk, the rest is textual."""
        usb = self._usb_ancestor(resolved)
        if usb is not None:
            return "usb", usb
        joined = "/".join(resolved.parts)
        if "/nvme/" in joined:
            return "nvme", None
        if re.search(r"/ata\d*/", joined):
            return "ata", None
        if "/mmc" in joined:
            return "mmc", None
        if "/virtio" in joined:
            return "virtio", None
        return None, None

    def _disk(self, name):
        """Whole disk backing a kernel block-device name, partitions included."""
        try:
            resolved = (self.path("/sys/class/block") / name).resolve()
        except OSError:
            return None
        if not resolved.is_dir():
            return None
        # A partition is a child of its whole disk inside the same block directory.
        if resolved.parent.parent.name == "block":
            return resolved.parent
        return resolved

    def block_devices(self):
        """Physical block devices: transport, size, and identity the kernel exposes.

        Virtual devices (loop, ram, device-mapper, md, optical) carry no transport
        and belong to their backing store, so they are skipped rather than reported
        with a fabricated bus.
        """
        base = self.path("/sys/block")
        if not base.is_dir():
            raise FileNotFoundError("block sysfs unavailable")
        result, failures = [], 0
        for disk in sorted(base.iterdir()):
            if re.match(r"(?:loop|ram|zram|dm-|md|sr|nbd|fd)", disk.name):
                continue
            try:
                resolved = disk.resolve()
            except OSError:
                failures += 1
                continue
            transport, usb = self._transport(resolved)
            labels = {"device": disk.name}
            if transport:
                labels["transport"] = transport
            if usb is not None:
                labels["usb_path"] = usb.name
            result.append(sample("spark_block_device_info", 1, **labels))
            for attribute, metric in (("removable", "removable"), ("queue/rotational", "rotational")):
                try:
                    value = int((disk / attribute).read_text().strip())
                    result.append(sample("spark_block_device_" + metric, value == 1, device=disk.name))
                except (OSError, ValueError):
                    failures += 1
            try:
                sectors = int((disk / "size").read_text().strip())
                result.append(sample("spark_block_device_size_bytes", sectors * 512, device=disk.name))
            except (OSError, ValueError):
                failures += 1
            for attribute, kind in (("device/model", "model"), ("device/vendor", "vendor"),
                                    ("device/serial", "serial"), ("device/wwid", "wwid")):
                source = disk / attribute
                # Kernel builds differ in which identity attributes they publish;
                # absence is not a read failure.
                if not source.is_file():
                    continue
                try:
                    value = source.read_text().strip()
                    if value:
                        result.append(sample("spark_block_device_identity", 1,
                                             kind=kind, detail=value, device=disk.name))
                except OSError:
                    failures += 1
        result.append(sample("spark_block_device_read_errors", failures))
        return "".join(result), failures == 0

    def usb(self):
        """USB devices with the port path that identifies them across replugs."""
        base = self.path("/sys/bus/usb/devices")
        if not base.is_dir():
            raise FileNotFoundError("USB sysfs unavailable")
        result, failures = [], 0
        for entry in sorted(base.iterdir()):
            # Root hubs are the host controllers themselves; PCI and NIC
            # collectors already describe that hardware.
            if entry.name.startswith("usb"):
                continue
            # Interface nodes (2-1:1.0) expose no device identity.
            if not (entry / "idVendor").is_file():
                continue
            labels = {"path": entry.name}
            try:
                labels["bus"] = (entry / "busnum").read_text().strip()
                labels["address"] = (entry / "devnum").read_text().strip()
                labels["id_vendor"] = (entry / "idVendor").read_text().strip()
                labels["id_product"] = (entry / "idProduct").read_text().strip()
            except OSError:
                failures += 1
                continue
            for attribute, label in (("manufacturer", "manufacturer"), ("product", "product"),
                                     ("serial", "serial")):
                try:
                    value = (entry / attribute).read_text().strip()
                    if value:
                        labels[label] = value
                except OSError:
                    pass
            result.append(sample("spark_usb_device_info", 1, **labels))
            try:
                # sysfs reports negotiated speed in Mbit/s.
                speed = number((entry / "speed").read_text().strip())
                if speed is not None:
                    result.append(sample("spark_usb_device_speed_bits_per_second",
                                         speed * 1e6, path=entry.name))
            except (OSError, ValueError):
                failures += 1
        result.append(sample("spark_usb_read_errors", failures))
        return "".join(result), failures == 0

    def external_filesystems(self):
        """Mounts backed by USB or removable media, with their bus attribution.

        Capacity stays in node_exporter's node_filesystem_* families for the same
        mountpoint; this family adds the transport identity those families lack.
        """
        result, failures, seen = [], 0, set()
        try:
            lines = self.path("/proc/self/mounts").read_text().splitlines()
        except OSError:
            raise FileNotFoundError("mount table unavailable")
        for line in lines:
            fields = line.split()
            if len(fields) < 3:
                continue
            source, mountpoint, fstype = fields[0], unescape_mount(fields[1]), fields[2]
            if not source.startswith("/dev/"):
                continue
            disk = self._disk(source.removeprefix("/dev/"))
            if disk is None:
                continue
            transport, _ = self._transport(disk.resolve())
            try:
                removable = int((disk / "removable").read_text().strip()) == 1
            except (OSError, ValueError):
                removable = False
            if transport != "usb" and not removable:
                continue
            key = (source, mountpoint)
            if key in seen:
                continue
            seen.add(key)
            labels = {"device": source, "mountpoint": mountpoint, "fstype": fstype}
            if transport:
                labels["transport"] = transport
            result.append(sample("spark_external_filesystem_info", 1, **labels))
            result.append(sample("spark_external_filesystem_removable", removable, **labels))
        result.append(sample("spark_external_filesystem_read_errors", failures))
        return "".join(result), failures == 0

    def rdma(self):
        base = self.path("/sys/class/infiniband")
        if not base.is_dir():
            raise FileNotFoundError("RDMA sysfs unavailable")
        result, failures = [], 0
        for device in sorted(base.iterdir()):
            for port in sorted((device / "ports").glob("*")):
                labels = {"device": device.name, "port": port.name}
                for field in ("state", "phys_state"):
                    try:
                        value = int((port / field).read_text().split(":", 1)[0])
                        result.append(sample("spark_rdma_" + field, value, **labels))
                    except (OSError, ValueError):
                        failures += 1
                try:
                    value = number((port / "rate").read_text().strip())
                    if value is None:
                        raise ValueError("no rate")
                    result.append(sample("spark_rdma_rate_bits_per_second", value * 1e9, **labels))
                except (OSError, ValueError):
                    failures += 1
                for group in ("counters", "hw_counters"):
                    for counter in sorted((port / group).glob("*")):
                        try:
                            raw = (counter.read_text().strip())
                            if not raw.isdigit():
                                raise ValueError("not a kernel counter")
                            unit = "4_bytes" if group == "counters" and counter.name in ("port_rcv_data", "port_xmit_data") else "raw"
                            result.append(sample("spark_rdma_counter_total", int(raw), counter=counter.name,
                                                 source=group, unit=unit, **labels))
                        except (OSError, ValueError):
                            failures += 1
        result.append(sample("spark_rdma_read_errors", failures))
        return "".join(result), failures == 0

    def physical_nics(self):
        base = self.path("/sys/class/net")
        if not base.is_dir():
            raise FileNotFoundError("network sysfs unavailable")
        return [interface for interface in sorted(base.iterdir()) if (interface / "device").is_dir()]

    def nics(self):
        result = []
        for interface in self.physical_nics():
            labels = {"device": interface.name}
            device = interface / "device"
            info = dict(labels)
            try:
                for line in (device / "uevent").read_text().splitlines():
                    key, _, value = line.partition("=")
                    if key == "DRIVER":
                        info["driver"] = value
                    elif key == "PCI_SLOT_NAME":
                        info["bus"] = value
            except OSError:
                pass
            if "driver" not in info and (device / "driver").is_symlink():
                try:
                    info["driver"] = (device / "driver").resolve(strict=True).name
                except OSError:
                    pass
            if "bus" not in info and re.fullmatch(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", device.resolve().name):
                info["bus"] = device.resolve().name
            result.append(sample("spark_nic_info", 1, **info))
            for field, metric, scale in (
                    ("speed", "speed_bits_per_second", 1e6), ("carrier", "carrier", 1),
                    ("mtu", "mtu_bytes", 1),
                    ("device/current_link_speed", "pcie_current_link_speed_transfers_per_second", 1e9),
                    ("device/max_link_speed", "pcie_max_link_speed_transfers_per_second", 1e9),
                    ("device/current_link_width", "pcie_current_link_width_lanes", 1),
                    ("device/max_link_width", "pcie_max_link_width_lanes", 1)):
                try:
                    value = number((interface / field).read_text().strip())
                    if value is not None and value >= 0:
                        result.append(sample("spark_nic_" + metric, value * scale, **labels))
                except (OSError, ValueError):
                    pass
            for field, states in (("operstate", {"unknown", "notpresent", "down", "lowerlayerdown", "testing", "dormant", "up"}),
                                  ("duplex", {"half", "full"})):
                try:
                    state = (interface / field).read_text().strip()
                    if state in states:
                        result.append(sample("spark_nic_" + field, 1, state=state, **labels))
                except OSError:
                    pass
            # Kernel interface counters for every physical NIC, independent of
            # the RDMA association: drops, errors and collisions expose fabric
            # trouble that byte counters alone hide.
            for field, metric in (("rx_errors", "receive_errors_total"),
                                  ("tx_errors", "transmit_errors_total"),
                                  ("rx_dropped", "receive_dropped_total"),
                                  ("tx_dropped", "transmit_dropped_total"),
                                  ("collisions", "collisions_total"),
                                  ("rx_crc_errors", "receive_crc_errors_total"),
                                  ("rx_frame_errors", "receive_frame_errors_total"),
                                  ("rx_missed_errors", "receive_missed_errors_total"),
                                  ("tx_carrier_errors", "transmit_carrier_errors_total"),
                                  ("rx_length_errors", "receive_length_errors_total"),
                                  ("rx_over_errors", "receive_over_errors_total")):
                try:
                    raw = (interface / "statistics" / field).read_text().strip()
                    if raw.isdigit():
                        result.append(sample("spark_nic_" + metric, int(raw), **labels))
                except OSError:
                    pass
        return "".join(result)

    def ethtool(self):
        # Physical devices plus RDMA associations, never arbitrary VPN/virtual netdevs.
        try:
            devices = {interface.name for interface in self.physical_nics()}
        except FileNotFoundError:
            devices = set()
        for directory in self.path("/sys/class/infiniband").glob("*/device/net"):
            devices.update(path.name for path in directory.iterdir())
        if not devices:
            return "", True
        result, failures = [], 0
        def collect(device):
            try:
                body = self.command(["ethtool", "-S", device])
                values = []
                for line in body.splitlines():
                    match = re.fullmatch(r"\s*([a-zA-Z0-9_.-]+):\s*(\d+)\s*", line)
                    if match:
                        values.append(sample("spark_ethtool_stat", int(match[2]), device=device, stat=match[1]))
                return "".join(values), bool(values)
            except (OSError, subprocess.SubprocessError):
                return "", False
        with ThreadPoolExecutor(max_workers=4) as pool:
            for body, ok in pool.map(collect, sorted(devices)):
                result.append(body)
                failures += not ok
        return "".join(result), failures == 0

    def supervisor(self):
        if not self.enable_supervisor:
            raise ValueError("supervisor collection is disabled")
        body, _ = self.fetch(8090, "/v1/status")
        status = json.loads(body)
        if not isinstance(status.get("lanes"), dict):
            raise ValueError("unrecognized supervisor status schema")
        result = []
        for key, metric, scale in (("memory_budget_mb", "memory_budget_bytes", 1024 ** 2),
                                   ("memory_used_mb", "memory_reserved_bytes", 1024 ** 2),
                                   ("lease_count", "leases", 1), ("auto_evict", "auto_evict", 1)):
            if key in status:
                result.append(sample("spark_supervisor_" + metric, status[key] * scale))
        for lane, state in status["lanes"].items():
            labels = {"lane": lane, "model": state["alias"]}
            result.append(sample("spark_supervisor_lane_ready", state["state"] == "ready", **labels))
            result.append(sample("spark_supervisor_lane_state", 1, state=state["state"], **labels))
            for key, metric, scale in (("reserved", "reserved", 1), ("draining", "draining", 1),
                                       ("leases", "leases", 1), ("memory_mb", "memory_declared_bytes", 1024 ** 2),
                                       ("unit_found", "unit_found", 1), ("port", "port", 1),
                                       ("priority", "priority", 1), ("elapsed_seconds", "elapsed_seconds", 1)):
                value = state.get(key, False) if key in ("reserved", "draining") else state.get(key)
                if value is not None:
                    result.append(sample("spark_supervisor_lane_" + metric, value * scale, **labels))
            result.append(sample("spark_supervisor_lane_unit_check_error", bool(state.get("unit_check_error")), **labels))
            # Provenance is descriptive identity, not a metric value.
            for key in ("display_name", "unit"):
                value = state.get(key)
                if isinstance(value, str) and value:
                    result.append(sample("spark_supervisor_lane_identity", 1, kind=key, detail=value, **labels))
            provenance = state.get("provenance")
            if isinstance(provenance, dict):
                for key in ("source", "artifact", "runtime", "revision"):
                    value = provenance.get(key)
                    if isinstance(value, str) and value:
                        result.append(sample("spark_supervisor_lane_identity", 1, kind=key, detail=value, **labels))
            fences = state.get("fences")
            if isinstance(fences, list):
                result.append(sample("spark_supervisor_lane_fence_count", len(fences), **labels))
        ready_aliases = status.get("ready_aliases")
        if isinstance(ready_aliases, list):
            result.append(sample("spark_supervisor_ready_alias_count", len(ready_aliases)))
        return "".join(result)

    def engine_stats(self):
        """Normalized per-endpoint throughput, queues, cache and latency.

        Derived from native counters the engines actually expose. Rates use the
        exporter's own observation interval, so a fresh exporter or a restarted
        engine reports no rate instead of a bogus spike. Percentiles are left to
        Prometheus, which needs the histogram buckets.
        """
        if not self.enable_inference:
            return ""
        now = time.time()
        result = []
        with ThreadPoolExecutor(max_workers=8) as pool:
            endpoints = list(pool.map(self._engine_stats_endpoint, sorted(self.ports())))
        for port, body in zip(sorted(self.ports()), endpoints):
            if body is None:
                continue
            labels = {"engine_port": str(port)}
            samples = parse_metrics(body)
            # Identity comes from the engine's own labels, never from a process name.
            model = next((labels_["model_name"] for name, labels_, _ in samples
                          if "model_name" in labels_), None)
            if model is None:
                model = next((labels_["model"] for name, labels_, _ in samples if "model" in labels_), None)
            if model:
                labels["model"] = model
            names = {sample_name for sample_name, _, _ in samples}
            engine = next((name for name, prompt, generation, _, _ in ENGINE_TOKEN_COUNTERS
                           if prompt in names or generation in names), None)
            if engine is None:
                engine = next((name for name, family, _ in ENGINE_GAUGES if family in names), None)
            if engine is None:
                engine = next((name for name, family, _, _ in ENGINE_CACHE if family in names), None)
            if engine is None:
                continue
            labels["engine"] = engine
            result.append(sample("spark_engine_endpoint_up", 1, **labels))

            for name, prompt_family, generation_family, _, selectors in ENGINE_TOKEN_COUNTERS:
                if name != engine:
                    continue
                for family, direction in ((prompt_family, "input"), (generation_family, "output")):
                    value = self._counter(samples, family, selectors)
                    if value is None:
                        continue
                    key = (str(port), family)
                    with self.engine_lock:
                        previous = self.engine_counters.get(key)
                        self.engine_counters[key] = (value, now)
                    result.append(sample("spark_engine_tokens_total", value,
                                         direction=direction, **labels))
                    if previous is not None:
                        last_value, last_time = previous
                        elapsed = now - last_time
                        # A decreased counter means the engine restarted; the
                        # delta is not a real rate.
                        if elapsed > 0 and value >= last_value:
                            result.append(sample("spark_engine_tokens_per_second",
                                                 (value - last_value) / elapsed,
                                                 direction=direction, **labels))
                        else:
                            result.append(sample("spark_engine_counter_reset_total", 1,
                                                 direction=direction, **labels))

            for name, family, metric in ENGINE_GAUGES:
                if name != engine:
                    continue
                value = self._counter(samples, family)
                if value is not None:
                    result.append(sample("spark_engine_" + metric, value, **labels))

            for name, family, metric, scale in ENGINE_CACHE:
                if name != engine:
                    continue
                value = self._counter(samples, family)
                if value is not None:
                    result.append(sample("spark_engine_" + metric, value * scale, **labels))

            for name, family, metric in ENGINE_LATENCY:
                if name != engine:
                    continue
                total = self._counter(samples, family + "_sum")
                count = self._counter(samples, family + "_count")
                if total is not None and count:
                    result.append(sample("spark_engine_" + metric + "_mean", total / count, **labels))
                if count is not None:
                    result.append(sample("spark_engine_" + metric + "_observations_total", count, **labels))

            for name, family, metric in ENGINE_SPECULATION:
                if name != engine:
                    continue
                value = self._counter(samples, family)
                if value is not None:
                    result.append(sample("spark_engine_" + metric, value, **labels))
            drafts = next((v for n, f, m in ENGINE_SPECULATION if m == "mtp_draft_tokens_total" and n == engine
                           for v in [self._counter(samples, f)] if v is not None), None)
            accepted = next((v for n, f, m in ENGINE_SPECULATION if m == "mtp_accepted_tokens_total" and n == engine
                             for v in [self._counter(samples, f)] if v is not None), None)
            if drafts and accepted is not None:
                result.append(sample("spark_engine_speculation_acceptance_ratio", accepted / drafts, **labels))
        return "".join(result)

    @staticmethod
    def _counter(samples, family, selectors=None):
        """Sum one exact native family, optionally filtered by labels.

        A family can appear once per worker/partition; summing raw counters keeps
        the endpoint total honest. Missing families return None, never zero.
        """
        total = None
        for name, labels, value in samples:
            if name != family:
                continue
            if selectors and any(labels.get(key) != expected for key, expected in selectors.items()):
                continue
            total = value if total is None else total + value
        return total

    def _engine_stats_endpoint(self, port):
        try:
            return self._engine_metrics(port)[0]
        except (OSError, ValueError, UnicodeError, http.client.HTTPException):
            return None

    def node_metrics(self):
        if self.node_exporter_port is None:
            raise ValueError("node exporter integration is disabled")
        body, _ = validate_metrics(*self.fetch(self.node_exporter_port, "/metrics"))
        return body

    def metrics(self):
        return self.flights.call("metrics", self._metrics)

    def _metrics(self):
        collectors = {"gpu": self.gpu, "gpu_processes": self.gpu_processes, "memory": self.memory,
                      "thermal": self.thermal, "cpu": self.cpu, "nvme": self.nvme,
                      "rdma": self.rdma, "ethtool": self.ethtool, "nics": self.nics,
                      "block_devices": self.block_devices, "usb": self.usb,
                      "external_filesystems": self.external_filesystems}
        if self.enable_supervisor:
            collectors["supervisor"] = self.supervisor
        if self.enable_inference:
            collectors["engine_stats"] = self.engine_stats
        if self.node_exporter_port is not None:
            collectors["node_exporter"] = self.node_metrics
        def collect(item):
            name, function = item
            start = time.monotonic()
            try:
                value = function()
                body, success = value if isinstance(value, tuple) else (value, True)
            except (OSError, ValueError, KeyError, TypeError, ET.ParseError, subprocess.SubprocessError, http.client.HTTPException):
                body, success = "", False
            now = time.time()
            with self.health_lock:
                if success:
                    self.last_success[name] = now
                last = self.last_success.get(name, 0)
            if isinstance(body, str):
                body = body.encode()
            separator = b"\n" if body and not body.endswith(b"\n") else b""
            health = sample("spark_collector_success", success, collector=name) + sample(
                "spark_collector_duration_seconds", time.monotonic() - start, collector=name) + sample(
                "spark_collector_last_success_timestamp_seconds", last, collector=name)
            return body + separator + health.encode()
        with ThreadPoolExecutor(max_workers=len(collectors)) as pool:
            parts = list(pool.map(collect, collectors.items()))
        return (b"# TYPE spark_rdma_counter_total counter\n" +
                sample("spark_host_info", 1, version=VERSION, architecture=platform.machine()).encode() + b"".join(parts))

    def ports(self):
        if not self.enable_inference:
            return set()
        ports = set(range(self.worker_range[0], self.worker_range[1] + 1)) | COMMON_ENGINE_PORTS | self.engine_endpoints.keys()
        configs = sorted(self.config_dir.glob("*.conf")) if self.enable_supervisor else ()
        for config in configs:
            try:
                for line in config.read_text().splitlines():
                    match = re.fullmatch(r"\s*PORT\s*=\s*['\"]?([0-9]+)['\"]?\s*", line)
                    if match:
                        port = int(match[1])
                        if 1 <= port <= 65535:
                            ports.add(port)
            except OSError:
                continue
        return ports - {8090, 9100, 9400, self.node_exporter_port}

    def engine_metrics(self, port):
        if port not in self.ports():
            raise ValueError("not a configured inference port")
        return self._engine_metrics(port)

    def _engine_metrics(self, port):
        native_path, standard_path = ENGINE_METRICS_PATHS
        if port in self.engine_endpoints:
            return validate_metrics(*self.fetch(port, self.engine_endpoints[port]))
        try:
            response = self.fetch(port, native_path)
        except HTTPStatusError as error:
            if error.status != 404:
                raise
        else:
            # A present native endpoint, even empty/invalid, must never trigger
            # a diagnostic /metrics request on TensorRT-LLM.
            return validate_metrics(*response)
        try:
            body, content_type = self.fetch(port, "/openapi.json")
        except HTTPStatusError as error:
            if error.status != 404:
                raise
        else:
            media_type = content_type.split(";", 1)[0].strip().lower()
            if media_type != "application/json" and not media_type.endswith("+json"):
                raise ValueError("unrecognized engine OpenAPI content type")
            schema = json.loads(body)
            if not isinstance(schema, dict) or not isinstance(schema.get("paths"), dict) or not isinstance(
                    schema.get("openapi", schema.get("swagger")), str):
                raise ValueError("unrecognized engine OpenAPI schema")
            if "/prometheus/metrics" in schema["paths"]:
                raise ValueError("native engine monitoring is unavailable")
            route = schema["paths"].get("/metrics", {})
            produces = schema.get("produces", [])
            if not isinstance(route, dict) or not isinstance(produces, list):
                raise ValueError("unrecognized engine metrics schema")
            serialized = json.dumps(route).lower()
            if "$ref" in serialized or "application/json" in serialized or "+json" in serialized or "iteration" in serialized or any(
                    "json" in str(media_type).lower() for media_type in produces):
                raise ValueError("engine /metrics is a diagnostic endpoint")
        return validate_metrics(*self.fetch(port, standard_path))

    def targets(self):
        if not self.enable_inference:
            return []
        return self.flights.call("targets", self._targets)

    def _targets(self):
        def probe(port):
            try:
                self._engine_metrics(port)
                return {"targets": [f"{self.address}:9400"], "labels": {
                    "__metrics_path__": f"/inference/{port}/metrics", "engine_port": str(port)}}
            except (OSError, ValueError, UnicodeError, http.client.HTTPException):
                return None
        with ThreadPoolExecutor(max_workers=8) as pool:
            return [group for group in pool.map(probe, sorted(self.ports())) if group is not None]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # Do not log request paths, caller headers, or upstream payloads.

    def do_GET(self):
        try:
            if self.path == "/metrics":
                body, content_type = self.server.exporter.metrics(), PROM_CONTENT_TYPE
            elif self.path == "/targets":
                body, content_type = json.dumps(self.server.exporter.targets()).encode(), "application/json"
            elif match := re.fullmatch(r"/inference/([1-9][0-9]{0,4})/metrics", self.path):
                port = int(match[1])
                if not self.server.exporter.enable_inference or port > 65535:
                    self.send_error(404)
                    return
                body, content_type = self.server.exporter.engine_metrics(port)
            else:
                self.send_error(404)
                return
        except (OSError, ValueError, UnicodeError, http.client.HTTPException):
            self.send_error(502, "Metrics endpoint unavailable")
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=VERSION)
    parser.add_argument("--listen-address", default="127.0.0.1",
                        help="concrete IPv4 literal or 'tailscale' (default: 127.0.0.1)")
    parser.add_argument("--enable-supervisor", action="store_true", help="read model supervisor status and lane configuration")
    parser.add_argument("--enable-inference", action="store_true", help="enable bounded loopback metrics discovery and proxy")
    parser.add_argument("--node-exporter-url", help="optional private http://127.0.0.1:<port>/metrics endpoint")
    parser.add_argument("--engine-endpoint", action="append", default=[], metavar="PORT:PATH",
                        help="explicit known-safe native endpoint; PATH is /metrics or /prometheus/metrics (repeatable)")
    parser.add_argument("--root", default="/", help="hardware fixture root (production: /)")
    parser.add_argument("--config-dir", default="/etc/spark-exporter/lanes.d")
    parser.add_argument("--worker-port-range", default="8000:8031")
    args = parser.parse_args()
    try:
        low, high = (int(value) for value in args.worker_port_range.split(":"))
        exporter = Exporter(args.listen_address, args.root, args.config_dir, (low, high),
                            enable_supervisor=args.enable_supervisor, enable_inference=args.enable_inference,
                            node_exporter_url=args.node_exporter_url, engine_endpoints=args.engine_endpoint)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    server = ThreadingHTTPServer((exporter.address, 9400), Handler)
    server.exporter = exporter
    server.serve_forever()


if __name__ == "__main__":
    main()
