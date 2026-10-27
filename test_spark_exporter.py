# SPDX-License-Identifier: MIT
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

import spark_exporter as exporter

GPU_XML = """<nvidia_smi_log><gpu><uuid>GPU-fixture</uuid><product_name>NVIDIA GB10</product_name>
<temperature><gpu_temp>52 C</gpu_temp></temperature><performance_state>P0</performance_state>
<gpu_power_readings><instant_power_draw>61.5 W</instant_power_draw><average_power_draw>60 W</average_power_draw><current_power_limit>N/A</current_power_limit></gpu_power_readings>
<clocks><graphics_clock>1200 MHz</graphics_clock><sm_clock>1200 MHz</sm_clock><mem_clock>4267 MHz</mem_clock></clocks>
<fb_memory_usage><total>N/A</total><used>N/A</used><free>N/A</free></fb_memory_usage>
<utilization><gpu_util>N/A</gpu_util><memory_util>N/A</memory_util></utilization>
<clocks_event_reasons><sw_power_cap>Active</sw_power_cap><hw_thermal_slowdown>Not Active</hw_thermal_slowdown></clocks_event_reasons>
</gpu></nvidia_smi_log>"""
HISTOGRAM = b'''# HELP vllm:request_latency_seconds Latency
# TYPE vllm:request_latency_seconds histogram
vllm:request_latency_seconds_bucket{model_name="fixture",le="0.1"} 3
vllm:request_latency_seconds_bucket{model_name="fixture",le="+Inf"} 5
vllm:request_latency_seconds_sum{model_name="fixture"} 1.23
vllm:request_latency_seconds_count{model_name="fixture"} 5
'''


class ExporterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = self.root / "models.d"
        self.config.mkdir()
        self.responses = {}
        self.calls = []
        self.subject = exporter.Exporter("100.100.1.2", self.root, self.config, (8000, 8002),
                                         command=self.command, fetch=self.fetch)

    def put(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)

    def link(self, name, target):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(self.root / target)
        return path

    def command(self, args):
        self.calls.append(args)
        if args == ["nvidia-smi", "-q", "-x"]:
            return GPU_XML
        if args[0] == "nvidia-smi":
            return "GPU-fixture, 1234, 91000\nGPU-fixture, 5678, N/A\n"
        if args == ["ethtool", "-S", "roce0"]:
            return "NIC statistics:\n tx_prio0_pause: 7\n rx_prio3_pause_duration: 150\n rx_bytes: 12345\n"
        raise FileNotFoundError(args[0])

    def fetch(self, port, path):
        self.calls.append((port, path))
        value = self.responses.get((port, path))
        if value is None:
            if any(candidate == port for candidate, _ in self.responses):
                raise exporter.HTTPStatusError(404)
            raise ConnectionRefusedError()
        if isinstance(value, Exception):
            raise value
        return value

    def test_gpu_supported_fields_not_fake_zero(self):
        metrics = self.subject.gpu()
        self.assertIn('spark_gpu_temperature_celsius{gpu="0",uuid="GPU-fixture"} 52.0', metrics)
        self.assertIn('spark_gpu_power_watts{gpu="0",uuid="GPU-fixture"} 61.5', metrics)
        self.assertIn('spark_gpu_clock_sm_hertz{gpu="0",uuid="GPU-fixture"} 1200000000.0', metrics)
        self.assertIn('field="utilization_ratio",gpu="0",uuid="GPU-fixture"} 0', metrics)
        self.assertNotIn("spark_gpu_utilization_ratio{", metrics)
        self.assertNotIn("spark_gpu_memory_total_bytes{", metrics)
        self.assertIn('reason="sw_power_cap",uuid="GPU-fixture"} 1', metrics)
        self.assertIn('spark_gpu_pstate{gpu="0",uuid="GPU-fixture"} 0', metrics)

    def test_process_memory_without_command_names(self):
        metrics = self.subject.gpu_processes()
        self.assertIn('spark_gpu_process_memory_bytes{pid="1234",uuid="GPU-fixture"} 95420416000.0', metrics)
        self.assertNotIn('spark_gpu_process_memory_bytes{pid="5678"', metrics)
        self.assertNotIn("process_name", str(self.calls))

    def test_unified_memory_available_is_not_free_or_gpu_total(self):
        self.put("proc/meminfo", "MemTotal: 1000 kB\nMemFree: 100 kB\nMemAvailable: 250 kB\nCached: 200 kB\nSwapFree: 0 kB\n")
        metrics = self.subject.memory()
        self.assertIn("spark_system_memory_available_bytes 256000", metrics)
        self.assertIn("spark_system_memory_unavailable_bytes 768000", metrics)
        self.assertNotIn("spark_gpu", metrics)

    def test_raw_rdma_words_and_mlx5_counters_and_partial_error(self):
        port = "sys/class/infiniband/mlx5_0/ports/1/"
        self.put(port + "state", "4: ACTIVE\n")
        self.put(port + "phys_state", "5: LinkUp\n")
        self.put(port + "rate", "200 Gb/sec (4X HDR)\n")
        self.put(port + "counters/port_rcv_data", "12345\n")
        self.put(port + "hw_counters/rx_write_requests", "987654321\n")
        self.put(port + "hw_counters/unsupported", "N/A\n")
        metrics, success = self.subject.rdma()
        self.assertFalse(success)
        self.assertIn('counter="port_rcv_data",device="mlx5_0",port="1",source="counters",unit="4_bytes"} 12345', metrics)
        self.assertIn('counter="rx_write_requests",device="mlx5_0",port="1",source="hw_counters",unit="raw"} 987654321', metrics)
        self.assertIn('spark_rdma_rate_bits_per_second{device="mlx5_0",port="1"} 200000000000.0', metrics)
        self.assertNotIn('counter="unsupported"', metrics)
        self.assertIn("spark_rdma_read_errors 1", metrics)

    def test_optional_pfc_and_ethtool_permission_failure(self):
        (self.root / "sys/class/infiniband/mlx5_0/device/net/roce0").mkdir(parents=True)
        metrics, success = self.subject.ethtool()
        self.assertTrue(success)
        self.assertIn('stat="tx_prio0_pause"} 7', metrics)
        self.subject.command = lambda args: (_ for _ in ()).throw(PermissionError())
        self.assertEqual(self.subject.ethtool(), ("", False))

    def test_discovery_only_responsive_metrics_no_cache_and_no_local_targets(self):
        self.subject.enable_inference = True
        self.responses[(8000, "/metrics")] = (HISTOGRAM, exporter.PROM_CONTENT_TYPE)
        self.responses[(8001, "/metrics")] = (b"<html>login</html>", "text/html")
        groups = self.subject.targets()
        self.assertEqual(groups, [{"targets": ["100.100.1.2:9400"], "labels": {
            "__metrics_path__": "/inference/8000/metrics", "engine_port": "8000"}}])
        self.responses.clear()
        self.assertEqual(self.subject.targets(), [])
        self.responses[(8002, "/metrics")] = (HISTOGRAM, exporter.PROM_CONTENT_TYPE)
        self.assertEqual(self.subject.targets()[0]["labels"]["engine_port"], "8002")

    def test_safe_configuration_without_executing_shell(self):
        self.subject.enable_supervisor = True
        self.subject.enable_inference = True
        (self.config / "good.conf").write_text('PORT="8888"\nSTART_COMMAND=touch /must-not-run\n')
        (self.config / "disabled.conf.disabled").write_text("PORT=8889\n")
        (self.config / "bad.conf").write_text("PORT=$(touch /must-not-run)\nPORT=9400\nPORT=65536\nPORT=-1\n")
        self.assertEqual(self.subject.ports(), {8000, 8001, 8002, 8888} | exporter.COMMON_ENGINE_PORTS)
        with self.assertRaises(ValueError):
            self.subject.engine_metrics(22)

    def test_proxy_keeps_histogram_and_openmetrics_exemplars_verbatim(self):
        self.subject.enable_inference = True
        content = HISTOGRAM + b'vllm:tokens_total{model_name="fixture"} 5 # {trace_id="abc"} 2 123.5\n# EOF\n'
        self.responses[(8000, "/metrics")] = (content, "application/openmetrics-text; version=1.0.0")
        self.assertEqual(self.subject.engine_metrics(8000), self.responses[(8000, "/metrics")])
        for body, kind in ((b"secret", "text/plain"), (b"# comment only\n", "text/plain"), (b"metric 1\n", "text/html")):
            with self.assertRaises(ValueError):
                exporter.validate_metrics(body, kind)

    def test_supervisor_readiness_and_declared_not_measured_reservations(self):
        self.subject.enable_supervisor = True
        status = {"memory_budget_mb": 100000, "memory_used_mb": 22000, "lease_count": 2,
                  "auto_evict": False, "lanes": {"lane-a": {"alias": "fast", "state": "ready",
                  "memory_mb": 22000, "leases": 2, "unit_found": True}, "lane-b": {
                  "alias": "large", "state": "failed", "memory_mb": 90000, "leases": 0,
                  "unit_found": True, "reserved": True, "unit_check_error": "private error"}}}
        self.responses[(8090, "/v1/status")] = (json.dumps(status).encode(), "application/json")
        metrics = self.subject.supervisor()
        self.assertIn("spark_supervisor_memory_reserved_bytes 23068672000", metrics)
        self.assertIn('spark_supervisor_lane_ready{lane="lane-b",model="large"} 0', metrics)
        self.assertIn('spark_supervisor_lane_reserved{lane="lane-b",model="large"} 1', metrics)
        self.assertNotIn("private error", metrics)

    def test_failed_collectors_keep_health_but_not_stale_values(self):
        self.put("proc/meminfo", "MemTotal: 1000 kB\nMemAvailable: 250 kB\n")
        good = self.subject.metrics().decode()
        self.assertIn("spark_host_info", good)
        self.assertIn('spark_collector_success{collector="gpu"} 1', good)
        self.subject.command = lambda args: (_ for _ in ()).throw(subprocess.TimeoutExpired(args, 4))
        failed = self.subject.metrics().decode()
        self.assertIn('spark_collector_success{collector="gpu"} 0', failed)
        self.assertNotIn("spark_gpu_temperature_celsius", failed)
        self.assertGreater(self.subject.last_success["gpu"], 0)
        self.assertNotIn("spark_supervisor_lane_ready", failed)

    def test_subprocess_is_read_only_bounded_and_has_no_credentials(self):
        with patch("spark_exporter.subprocess.run") as run:
            run.return_value.stdout = GPU_XML
            exporter.run_command(["nvidia-smi", "-q", "-x"])
            args, kwargs = run.call_args
            self.assertEqual(args[0], ["nvidia-smi", "-q", "-x"])
            self.assertEqual(kwargs["timeout"], 4)
            self.assertEqual(set(kwargs["env"]), {"PATH", "LC_ALL"})
            self.assertNotIn("shell", kwargs)

    def test_literal_ipv4_binding_accepts_lan_and_loopback_not_wildcard(self):
        for address in ("127.0.0.1", "192.168.1.2", "100.100.1.2"):
            with self.subTest(address=address):
                subject = exporter.Exporter(address, command=self.command, fetch=self.fetch,
                                            worker_range=(8000, 8000), enable_inference=True)
                self.assertEqual(subject.address, address)
                self.responses[(8000, "/metrics")] = (HISTOGRAM, exporter.PROM_CONTENT_TYPE)
                self.assertEqual(subject.targets()[0]["targets"], [address + ":9400"])
        self.assertFalse(any(isinstance(call, list) for call in self.calls))
        for address in ("0.0.0.0", "example.net", "::1", "::", "239.1.2.3", "255.255.255.255"):
            with self.subTest(address=address), self.assertRaises(ValueError):
                exporter.Exporter(address)

    def test_tailscale_is_resolved_at_runtime_only_when_selected(self):
        with patch("spark_exporter.subprocess.run") as run:
            run.return_value.stdout = "100.100.1.2\n"
            self.assertEqual(exporter.Exporter("tailscale").address, "100.100.1.2")
            self.assertEqual(run.call_args.args[0], ["tailscale", "ip", "-4"])
            self.assertEqual(run.call_args.kwargs["timeout"], 4)
            for value in ("", "192.168.1.2\n", "100.100.1.2\n100.100.1.3\n"):
                run.return_value.stdout = value
                with self.subTest(value=value), self.assertRaises(ValueError):
                    exporter.Exporter("tailscale")

    def test_disabled_integrations_make_no_http_probes_or_config_reads(self):
        (self.config / "lane.conf").write_text("PORT=8888\n")
        self.put("proc/meminfo", "MemTotal: 1000 kB\n")
        metrics = self.subject.metrics().decode()
        self.assertNotIn("spark_supervisor_", metrics)
        self.assertNotIn('collector="supervisor"', metrics)
        with patch("spark_exporter.Path.glob", side_effect=AssertionError("must not read lane configuration")):
            self.assertEqual(self.subject.ports(), set())
            self.assertEqual(self.subject.targets(), [])
        with self.assertRaises(ValueError):
            self.subject.engine_metrics(8000)
        with self.assertRaises(ValueError):
            self.subject.supervisor()
        self.assertFalse(any(isinstance(call, tuple) for call in self.calls))

    def test_inference_only_uses_range_without_supervisor_config_or_status(self):
        self.subject.enable_inference = True
        (self.config / "lane.conf").write_text("PORT=8888\n")
        self.responses[(8000, "/metrics")] = (HISTOGRAM, exporter.PROM_CONTENT_TYPE)
        with patch("spark_exporter.Path.glob", side_effect=AssertionError("supervisor is disabled")):
            self.assertEqual(self.subject.ports(), {8000, 8001, 8002} | exporter.COMMON_ENGINE_PORTS)
            self.assertEqual(len(self.subject.targets()), 1)
        expected = {(port, "/prometheus/metrics") for port in self.subject.ports()} | {
            (8000, "/openapi.json"), (8000, "/metrics")}
        self.assertEqual(set(self.calls), expected)

    def test_supervisor_only_collects_status_but_does_not_probe_inference(self):
        self.subject.enable_supervisor = True
        self.responses[(8090, "/v1/status")] = (b'{"lanes": {}, "lease_count": 0}', "application/json")
        metrics = self.subject.metrics().decode()
        self.assertIn("spark_supervisor_leases 0", metrics)
        self.assertEqual(self.subject.targets(), [])
        self.assertEqual([call for call in self.calls if isinstance(call, tuple)], [(8090, "/v1/status")])

    def test_host_marker_uses_actual_architecture_not_gpu_product(self):
        with patch("spark_exporter.platform.machine", return_value="x86_64"):
            metrics = self.subject.metrics().decode()
        self.assertIn(f'spark_host_info{{architecture="x86_64",version="{exporter.VERSION}"}} 1', metrics)
        self.assertNotIn('architecture="gb10"', metrics)

    def test_cli_version_and_safe_defaults(self):
        with patch("sys.argv", ["spark-exporter", "--version"]), patch("sys.stdout", new_callable=io.StringIO) as stdout:
            with self.assertRaises(SystemExit) as stopped:
                exporter.main()
            self.assertEqual(stopped.exception.code, 0)
            self.assertEqual(stdout.getvalue(), exporter.VERSION + "\n")
        with patch("sys.argv", ["spark-exporter"]), patch("spark_exporter.ThreadingHTTPServer") as server:
            exporter.main()
            self.assertEqual(server.call_args.args[0], ("127.0.0.1", 9400))
            self.assertFalse(server.return_value.exporter.enable_supervisor)
            self.assertFalse(server.return_value.exporter.enable_inference)
            self.assertIsNone(server.return_value.exporter.node_exporter_port)

    def test_fetch_closes_response_and_connection_on_success_and_failures(self):
        for status, body, error in ((200, HISTOGRAM, None), (302, b"", ValueError),
                                    (200, b"x" * 9, ValueError), (200, None, OSError)):
            with self.subTest(status=status, body=body), patch("spark_exporter.http.client.HTTPConnection") as connection:
                response = connection.return_value.getresponse.return_value
                response.__enter__.return_value = response
                response.status = status
                response.read.side_effect = OSError() if body is None else None
                response.read.return_value = body
                response.getheader.return_value = exporter.PROM_CONTENT_TYPE
                if error is None:
                    self.assertEqual(exporter.fetch_local(8000, "/metrics", len(HISTOGRAM)), (HISTOGRAM, exporter.PROM_CONTENT_TYPE))
                else:
                    with self.assertRaises(error):
                        exporter.fetch_local(8000, "/metrics", 8)
                response.__exit__.assert_called_once()
                connection.return_value.close.assert_called_once()

    def test_disabled_http_endpoints_do_not_probe_upstream(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), exporter.Handler)
        server.exporter = self.subject
        self.addCleanup(server.server_close)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        connection = http.client.HTTPConnection(*server.server_address)
        self.addCleanup(connection.close)
        for path, status, body in (("/targets", 200, b"[]"), ("/inference/8000/metrics", 404, None)):
            connection.request("GET", path)
            with connection.getresponse() as response:
                self.assertEqual(response.status, status)
                content = response.read()
                if body is not None:
                    self.assertEqual(content, body)
        self.assertEqual(self.calls, [])

    def test_http_paths_and_auth_not_forwarded(self):
        self.subject.enable_supervisor = True
        self.subject.enable_inference = True
        observed = []
        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                observed.append((self.path, self.headers.get("Authorization"), self.headers.get("Cookie")))
                self.send_response(200)
                self.send_header("Content-Type", exporter.PROM_CONTENT_TYPE)
                self.end_headers()
                self.wfile.write(HISTOGRAM)
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        self.addCleanup(upstream.server_close)
        worker = threading.Thread(target=upstream.serve_forever, daemon=True)
        worker.start()
        self.addCleanup(upstream.shutdown)
        port = upstream.server_address[1]
        (self.config / "fixture.conf").write_text(f"PORT={port}\n")
        self.subject.fetch = exporter.fetch_local
        server = ThreadingHTTPServer(("127.0.0.1", 0), exporter.Handler)
        server.exporter = self.subject
        self.addCleanup(server.server_close)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        connection = http.client.HTTPConnection(*server.server_address)
        self.addCleanup(connection.close)
        connection.request("GET", f"/inference/{port}/metrics", headers={"Authorization": "Bearer fixture-secret", "Cookie": "secret=fixture"})
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        self.assertEqual(response.read(), HISTOGRAM)
        self.assertEqual(observed, [("/prometheus/metrics", None, None)])
        for path in ("/inference/22/metrics", "/inference/9400/metrics", "/inference/65536/metrics", "/inference/08000/metrics", "/inference/8000/metrics?path=/v1/status", "/v1/status"):
            connection.request("GET", path)
            response = connection.getresponse()
            self.assertIn(response.status, (404, 502))
            response.read()
        self.assertEqual(len(observed), 1)

    def test_physical_200gb_nic_and_pcie_without_rdma_or_fixed_names(self):
        base = "sys/class/net/enp37s0f9/"
        for field, value in {
                "device/uevent": "DRIVER=mlx5_core\nPCI_SLOT_NAME=0000:25:00.0\n",
                "speed": "200000\n", "carrier": "1\n", "operstate": "up\n",
                "mtu": "9000\n", "duplex": "full\n", "device/current_link_speed": "16.0 GT/s PCIe\n",
                "device/max_link_speed": "32.0 GT/s PCIe\n", "device/current_link_width": "16\n",
                "device/max_link_width": "16\n"}.items():
            self.put(base + field, value)
        self.put("sys/class/net/tailscale0/speed", "1000000\n")
        metrics = self.subject.nics()
        self.assertIn('spark_nic_info{bus="0000:25:00.0",device="enp37s0f9",driver="mlx5_core"} 1', metrics)
        self.assertIn('spark_nic_speed_bits_per_second{device="enp37s0f9"} 200000000000.0', metrics)
        self.assertIn('spark_nic_pcie_current_link_speed_transfers_per_second{device="enp37s0f9"} 16000000000.0', metrics)
        self.assertIn('spark_nic_pcie_current_link_width_lanes{device="enp37s0f9"} 16.0', metrics)
        self.assertIn('spark_nic_operstate{device="enp37s0f9",state="up"} 1', metrics)
        self.assertIn('spark_nic_duplex{device="enp37s0f9",state="full"} 1', metrics)
        self.assertNotIn("tailscale0", metrics)
        self.assertFalse(self.calls)

    def test_nic_unknown_speed_and_pcie_are_not_fabricated_zero(self):
        self.put("sys/class/net/eth42/device/uevent", "")
        self.put("sys/class/net/eth42/speed", "-1\n")
        self.put("sys/class/net/eth42/carrier", "0\n")
        self.put("sys/class/net/eth42/duplex", "unknown\n")
        metrics = self.subject.nics()
        self.assertIn('spark_nic_carrier{device="eth42"} 0.0', metrics)
        self.assertNotIn("spark_nic_speed_", metrics)
        self.assertNotIn("spark_nic_pcie_", metrics)
        self.assertNotIn("spark_nic_duplex", metrics)

    def test_ethtool_all_physical_nics_preserve_pfc_cnp_ecn_errors(self):
        self.put("sys/class/net/enp37s0f9/device/uevent", "DRIVER=mlx5_core\n")
        self.put("sys/class/net/veth0/operstate", "up\n")
        self.subject.command = lambda args: self.calls.append(args) or (
            "NIC statistics:\n tx_prio7_pause: 7\n rx_prio3_pause_duration: 150\n"
            " np_cnp_sent: 31\n rp_cnp_handled: 42\n rx_ecn_mark: 10\n rx_crc_errors: 9\n")
        metrics, success = self.subject.ethtool()
        self.assertTrue(success)
        self.assertEqual(self.calls, [["ethtool", "-S", "enp37s0f9"]])
        for stat, value in (("tx_prio7_pause", 7), ("rx_prio3_pause_duration", 150),
                            ("np_cnp_sent", 31), ("rp_cnp_handled", 42),
                            ("rx_ecn_mark", 10), ("rx_crc_errors", 9)):
            self.assertIn(f'spark_ethtool_stat{{device="enp37s0f9",stat="{stat}"}} {value}', metrics)

    def test_gpu_pcie_only_when_driver_xml_reports_numeric_fields(self):
        pci = ("<pci><pci_gpu_link_info><pcie_gen><current_link_gen>4</current_link_gen>"
               "<max_link_gen>5</max_link_gen></pcie_gen><link_widths>"
               "<current_link_width>16x</current_link_width><max_link_width>N/A</max_link_width>"
               "</link_widths></pci_gpu_link_info></pci>")
        self.subject.command = lambda args: GPU_XML.replace("</gpu>", pci + "</gpu>")
        metrics = self.subject.gpu()
        self.assertIn('spark_gpu_pcie_generation{gpu="0",uuid="GPU-fixture"} 4.0', metrics)
        self.assertIn('spark_gpu_pcie_link_width_lanes{gpu="0",uuid="GPU-fixture"} 16.0', metrics)
        self.assertNotIn("spark_gpu_pcie_max_link_width_lanes{", metrics)
        self.assertIn('field="pcie_max_link_width_lanes",gpu="0",uuid="GPU-fixture"} 0', metrics)

    def test_node_exporter_merge_preserves_standard_metrics_and_health(self):
        raw = b"# TYPE node_cpu_seconds_total counter\nnode_cpu_seconds_total{cpu=\"0\",mode=\"idle\"} 123.456\n"
        subject = exporter.Exporter("127.0.0.1", self.root, command=self.command, fetch=self.fetch,
                                    node_exporter_url="http://127.0.0.1:19100/metrics")
        self.responses[(19100, "/metrics")] = (raw, exporter.PROM_CONTENT_TYPE)
        metrics = subject.metrics()
        self.assertIn(raw, metrics)
        self.assertIn(b'spark_collector_success{collector="node_exporter"} 1\n', metrics)
        self.assertEqual([call for call in self.calls if isinstance(call, tuple)], [(19100, "/metrics")])
        self.responses.clear()
        metrics = subject.metrics()
        self.assertNotIn(b"node_cpu_seconds_total", metrics)
        self.assertIn(b'spark_collector_success{collector="node_exporter"} 0\n', metrics)
        self.assertGreater(subject.last_success["node_exporter"], 0)

    def test_node_exporter_url_rejects_arbitrary_hosts_paths_and_self_recursion(self):
        for url in ("http://localhost:19100/metrics", "http://192.168.1.1:19100/metrics",
                    "http://127.0.0.2:19100/metrics", "https://127.0.0.1:19100/metrics",
                    "http://user:password@127.0.0.1:19100/metrics", "http://127.0.0.1:0/metrics",
                    "http://127.0.0.1:65536/metrics", "http://127.0.0.1:9400/metrics",
                    "http://127.0.0.1:19100/metrics?secret=yes", "http://127.0.0.1:19100/v1/status",
                    "http://127.0.0.1:19100/metrics#fragment"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                exporter.Exporter("127.0.0.1", node_exporter_url=url)
        self.assertEqual(self.calls, [])

    def test_node_exporter_is_not_discovered_as_inference_engine(self):
        subject = exporter.Exporter("127.0.0.1", worker_range=(19100, 19101),
                                    enable_inference=True, fetch=self.fetch,
                                    node_exporter_url="http://127.0.0.1:19100/metrics")
        self.assertEqual(subject.ports(), {19101} | exporter.COMMON_ENGINE_PORTS)
        self.assertEqual(subject.targets(), [])
        self.assertEqual(set(self.calls), {(port, "/prometheus/metrics") for port in subject.ports()})

    def test_common_engine_ports_and_tensorrt_preferred_path_without_draining_json(self):
        self.subject.enable_inference = True
        self.assertTrue({8080, 9931, 30000}.issubset(self.subject.ports()))
        self.responses[(8000, "/metrics")] = (b'{"status": "ready"}', "application/json")
        raw = b"trtllm_generation_tokens_total 135\n"
        self.responses[(8000, "/prometheus/metrics")] = (raw, exporter.PROM_CONTENT_TYPE)
        self.assertEqual(self.subject.engine_metrics(8000), (raw, exporter.PROM_CONTENT_TYPE))
        self.assertEqual(self.calls, [(8000, "/prometheus/metrics")])
        groups = self.subject.targets()
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["labels"]["__metrics_path__"], "/inference/8000/metrics")
        self.assertFalse(any(path not in exporter.ENGINE_METRICS_PATHS for _, path in self.calls))

    def test_worker_range_is_bounded_for_imported_api_too(self):
        for limits in ((0, 1), (8002, 8000), (1, 129), (65535, 65536)):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                exporter.Exporter("127.0.0.1", worker_range=limits)

    def test_tensorrt_schema_blocks_json_diagnostic_without_draining_counter(self):
        self.subject.enable_inference = True
        for paths in (
                {"/prometheus/metrics": {}},
                {"/metrics": {"get": {"responses": {"200": {"content": {"application/json": {}}}}}}},
                {"/metrics": {"get": {"description": "Return and drain iteration statistics"}}}):
            with self.subTest(paths=paths):
                drains = []
                self.calls.clear()
                def fetch(port, path):
                    self.calls.append((port, path))
                    if path == "/prometheus/metrics":
                        raise exporter.HTTPStatusError(404)
                    if path == "/openapi.json":
                        return json.dumps({"openapi": "3.1.0", "paths": paths}).encode(), "application/json"
                    drains.append(1)
                    return b'{"iteration_stats": []}', "application/json"
                self.subject.fetch = fetch
                with self.assertRaises(ValueError):
                    self.subject.engine_metrics(8000)
                self.assertEqual(drains, [])
                self.assertEqual(self.calls, [(8000, "/prometheus/metrics"), (8000, "/openapi.json")])

    def test_present_but_empty_invalid_or_auth_protected_native_endpoint_fails_closed(self):
        self.subject.enable_inference = True
        for response in ((b"", exporter.PROM_CONTENT_TYPE), (b"metric not_numeric\n", exporter.PROM_CONTENT_TYPE),
                         (b'{"stats":[]}', "application/json"), exporter.HTTPStatusError(401),
                         exporter.HTTPStatusError(403), exporter.HTTPStatusError(500)):
            with self.subTest(response=response):
                self.calls.clear()
                self.responses[(8000, "/prometheus/metrics")] = response
                self.responses[(8000, "/metrics")] = (HISTOGRAM, exporter.PROM_CONTENT_TYPE)
                with self.assertRaises(ValueError):
                    self.subject.engine_metrics(8000)
                self.assertEqual(self.calls, [(8000, "/prometheus/metrics")])

    def test_openapi_auth_errors_malformed_schema_or_transport_failure_fail_closed(self):
        self.subject.enable_inference = True
        for response in (exporter.HTTPStatusError(401), exporter.HTTPStatusError(403),
                         exporter.HTTPStatusError(500), ConnectionRefusedError(),
                         (b"<html>login</html>", "text/html"), (b"{}", "application/json"),
                         (b"not json", "application/json")):
            with self.subTest(response=response):
                self.calls.clear()
                self.responses[(8000, "/openapi.json")] = response
                self.responses[(8000, "/metrics")] = (HISTOGRAM, exporter.PROM_CONTENT_TYPE)
                with self.assertRaises((ValueError, OSError)):
                    self.subject.engine_metrics(8000)
                self.assertEqual(self.calls, [(8000, "/prometheus/metrics"), (8000, "/openapi.json")])

    def test_verified_standard_metrics_with_or_without_openapi_keeps_raw_bytes(self):
        self.subject.enable_inference = True
        text_schema = {"openapi": "3.0.0", "paths": {
            "/metrics": {"get": {"responses": {"200": {"content": {"text/plain": {}}}}}}}}
        for schema in (exporter.HTTPStatusError(404),
                       (b'{"openapi":"3.0.0","paths":{}}', "application/json"),
                       (json.dumps(text_schema).encode(), "application/json")):
            with self.subTest(schema=schema):
                self.calls.clear()
                self.responses[(8000, "/openapi.json")] = schema
                self.responses[(8000, "/metrics")] = (HISTOGRAM, exporter.PROM_CONTENT_TYPE)
                self.assertEqual(self.subject.engine_metrics(8000), (HISTOGRAM, exporter.PROM_CONTENT_TYPE))
                self.assertEqual(self.calls, [(8000, "/prometheus/metrics"), (8000, "/openapi.json"), (8000, "/metrics")])

    def test_explicit_endpoints_are_path_only_do_not_probe_schema_or_fallback(self):
        subject = exporter.Exporter("127.0.0.1", fetch=self.fetch, enable_inference=True,
                                    engine_endpoints=("8888:/metrics", "8889:/prometheus/metrics"))
        self.responses[(8888, "/metrics")] = (HISTOGRAM, exporter.PROM_CONTENT_TYPE)
        self.responses[(8889, "/prometheus/metrics")] = exporter.HTTPStatusError(404)
        self.assertIn(8888, subject.ports())
        self.assertEqual(subject.engine_metrics(8888), (HISTOGRAM, exporter.PROM_CONTENT_TYPE))
        with self.assertRaises(ValueError):
            subject.engine_metrics(8889)
        self.assertEqual(self.calls, [(8888, "/metrics"), (8889, "/prometheus/metrics")])
        for endpoints in (("8000:/v1/models",), ("8000:/metrics?key=secret",),
                          ("8000:http://127.0.0.1/metrics",), ("0:/metrics",),
                          ("65536:/metrics",), ("9400:/metrics",),
                          ("8000:/metrics", "8000:/prometheus/metrics")):
            with self.subTest(endpoints=endpoints), self.assertRaises(ValueError):
                exporter.Exporter("127.0.0.1", engine_endpoints=endpoints)


    def test_thermal_cpu_nvme_and_gpu_throttle_counters(self):
        self.put("sys/class/thermal/thermal_zone0/type", "acpitz\n")
        self.put("sys/class/thermal/thermal_zone0/temp", "87000\n")
        self.put("sys/class/hwmon/hwmon0/name", "mlx5\n")
        self.put("sys/class/hwmon/hwmon0/temp1_input", "73000\n")
        self.put("sys/class/hwmon/hwmon0/temp1_label", "asic\n")
        self.put("sys/class/hwmon/hwmon0/temp1_crit", "105000\n")
        self.put("sys/devices/system/cpu/cpufreq/policy0/scaling_governor", "schedutil\n")
        self.put("sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq", "338000\n")
        self.put("sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq", "2200000\n")
        self.put("sys/devices/system/cpu/cpufreq/policy0/cpuinfo_max_freq", "2808000\n")
        self.put("sys/devices/system/cpu/cpufreq/policy0/affected_cpus", "0 1 2 3\n")
        self.put("sys/class/nvme/nvme0/model", "FIXTURE-DISK\n")
        self.put("sys/class/nvme/nvme0/firmware_rev", "ERFM12.0\n")

        thermal, ok = self.subject.thermal()
        self.assertTrue(ok)
        self.assertIn('spark_thermal_zone_celsius{type="acpitz",zone="0"} 87.0', thermal)
        self.assertIn('spark_hwmon_temperature_celsius{chip="hwmon0:mlx5",label="asic",sensor="temp1"} 73.0', thermal)
        self.assertIn('spark_hwmon_temperature_critical_celsius{chip="hwmon0:mlx5",label="asic",sensor="temp1"} 105.0', thermal)

        cpu, ok = self.subject.cpu()
        self.assertTrue(ok)
        self.assertIn('spark_cpu_frequency_governor{cpu="0_1_2_3",governor="schedutil",policy="0"} 1', cpu)
        self.assertIn('spark_cpu_frequency_current_hertz{cpu="0_1_2_3",policy="0"} 338000000.0', cpu)
        self.assertIn('spark_cpu_frequency_hardware_max_hertz{cpu="0_1_2_3",policy="0"} 2808000000.0', cpu)

        nvme, ok = self.subject.nvme()
        self.assertTrue(ok)
        self.assertIn('spark_nvme_identity{controller="nvme0",detail="FIXTURE-DISK",kind="model"} 1', nvme)

    def test_block_devices_report_transport_and_skip_virtual_devices(self):
        usb_disk = "sys/devices/platform/NVDA8000:00/usb2/2-1/2-1:1.0/host0/target0:0:0/0:0:0:0/block/sda"
        self.put("sys/devices/platform/NVDA8000:00/usb2/2-1/idVendor", "0bda\n")
        for attribute, value in (("removable", "0\n"), ("queue/rotational", "1\n"),
                                 ("size", "4000797360\n"), ("device/model", "USB-DISK\n"),
                                 ("device/vendor", "Fixture\n")):
            self.put(usb_disk + "/" + attribute, value)
        self.put(usb_disk + "/sda1/partition", "1\n")
        nvme_disk = "sys/devices/pci0004:00/0004:01:00.0/nvme/nvme0/nvme0n1"
        for attribute, value in (("removable", "0\n"), ("queue/rotational", "0\n"), ("size", "1000204886\n")):
            self.put(nvme_disk + "/" + attribute, value)
        for name, target in (("sda", usb_disk), ("nvme0n1", nvme_disk)):
            self.link("sys/block/" + name, target)
        # Partitions live in /sys/class/block only; /sys/block holds whole disks.
        self.link("sys/class/block/sda1", usb_disk + "/sda1")
        self.put("sys/block/loop0/removable", "0\n")
        self.put("sys/block/dm-0/removable", "0\n")

        metrics, ok = self.subject.block_devices()
        self.assertTrue(ok)
        self.assertIn('spark_block_device_info{device="sda",transport="usb",usb_path="2-1"} 1', metrics)
        self.assertIn('spark_block_device_size_bytes{device="sda"} 2048408248320', metrics)
        self.assertIn('spark_block_device_rotational{device="sda"} 1', metrics)
        self.assertIn('spark_block_device_identity{detail="USB-DISK",device="sda",kind="model"} 1', metrics)
        self.assertIn('spark_block_device_info{device="nvme0n1",transport="nvme"} 1', metrics)
        # Partitions are not whole disks, and loop/device-mapper nodes have no bus.
        self.assertNotIn('device="sda1"', metrics)
        self.assertNotIn('device="loop0"', metrics)
        self.assertNotIn('device="dm-0"', metrics)
        self.assertIn("spark_block_device_read_errors 0", metrics)

    def test_usb_devices_skip_root_hubs_and_interfaces(self):
        self.put("sys/bus/usb/devices/usb1/idVendor", "1d6b\n")
        self.put("sys/bus/usb/devices/2-1:1.0/bInterfaceClass", "08\n")
        for attribute, value in (("idVendor", "0bda\n"), ("idProduct", "9220\n"),
                                 ("busnum", "2\n"), ("devnum", "2\n"),
                                 ("manufacturer", "Ugreen\n"), ("product", "Storage Device\n"),
                                 ("serial", "012345679031\n"), ("speed", "20000\n")):
            self.put("sys/bus/usb/devices/2-1/" + attribute, value)

        metrics, ok = self.subject.usb()
        self.assertTrue(ok)
        self.assertIn('spark_usb_device_info{address="2",bus="2",id_product="9220",id_vendor="0bda",'
                      'manufacturer="Ugreen",path="2-1",product="Storage Device",serial="012345679031"} 1', metrics)
        self.assertIn('spark_usb_device_speed_bits_per_second{path="2-1"} 20000000000.0', metrics)
        self.assertNotIn('path="usb1"', metrics)
        self.assertNotIn('path="2-1:1.0"', metrics)
        self.assertIn("spark_usb_read_errors 0", metrics)

    def test_external_filesystems_only_cover_usb_and_removable_mounts(self):
        usb_disk = "sys/devices/platform/NVDA8000:00/usb2/2-1/2-1:1.0/host0/target0:0:0/0:0:0:0/block/sda"
        self.put("sys/devices/platform/NVDA8000:00/usb2/2-1/idVendor", "0bda\n")
        self.put(usb_disk + "/removable", "0\n")
        self.put(usb_disk + "/sda1/partition", "1\n")
        removable = "sys/devices/platform/NVDA8000:00/usb3/3-1/host1/target1:0:0/1:0:0:0/block/sdb"
        self.put(removable + "/removable", "1\n")
        self.put(removable + "/sdb1/partition", "1\n")
        root = "sys/devices/pci0004:00/0004:01:00.0/nvme/nvme0/nvme0n1"
        self.put(root + "/removable", "0\n")
        self.put(root + "/nvme0n1p2/partition", "2\n")
        for name, target in (("sda", usb_disk), ("sda1", usb_disk + "/sda1"),
                             ("sdb", removable), ("sdb1", removable + "/sdb1"),
                             ("nvme0n1", root), ("nvme0n1p2", root + "/nvme0n1p2")):
            self.link("sys/class/block/" + name, target)
        self.put("proc/self/mounts",
                 "/dev/nvme0n1p2 / ext4 rw 0 0\n"
                 "/dev/sda1 /mnt/disk\\040one ext4 rw 0 0\n"
                 "/dev/sda1 /mnt/disk\\040one ext4 rw 0 0\n"
                 "/dev/sdb1 /media/stick vfat rw 0 0\n"
                 "tmpfs /run tmpfs rw 0 0\n")

        metrics, ok = self.subject.external_filesystems()
        self.assertTrue(ok)
        # Octal-escaped spaces are decoded, and a repeated mount appears once.
        self.assertEqual(metrics.count('mountpoint="/mnt/disk one"'), 2)
        self.assertIn('spark_external_filesystem_info{device="/dev/sda1",fstype="ext4",'
                      'mountpoint="/mnt/disk one",transport="usb"} 1', metrics)
        self.assertIn('spark_external_filesystem_removable{device="/dev/sda1",fstype="ext4",'
                      'mountpoint="/mnt/disk one",transport="usb"} 0', metrics)
        self.assertIn('spark_external_filesystem_info{device="/dev/sdb1",fstype="vfat",'
                      'mountpoint="/media/stick"} 1', metrics)
        self.assertIn('spark_external_filesystem_removable{device="/dev/sdb1",fstype="vfat",'
                      'mountpoint="/media/stick"} 1', metrics)
        # The root device is standard node_exporter territory.
        self.assertNotIn('mountpoint="/"', metrics)
        self.assertNotIn("tmpfs", metrics)
        self.assertIn("spark_external_filesystem_read_errors 0", metrics)

    def test_external_filesystems_fail_cleanly_without_mount_table(self):
        with self.assertRaises(FileNotFoundError):
            self.subject.external_filesystems()

    def test_gpu_temperature_limits_throttle_durations_and_identity(self):
        xml = """<nvidia_smi_log><gpu><uuid>GPU-x</uuid><product_name>NVIDIA GB10</product_name>
<driver_version>580.1</driver_version><cuda_version>13.0</cuda_version><vbios_version>9A.0B</vbios_version>
<gpu_part_number>2E12-275-A1</gpu_part_number><gsp_firmware_version>580.1</gsp_firmware_version>
<persistence_mode>Enabled</persistence_mode><c2c_mode>Enabled</c2c_mode>
<temperature><gpu_temp>68 C</gpu_temp><gpu_temp_tlimit>27 C</gpu_temp_tlimit>
<gpu_temp_max_tlimit_threshold>N/A</gpu_temp_max_tlimit_threshold><memory_temp>N/A</memory_temp></temperature>
<utilization><gpu_util>0 %</gpu_util><encoder_util>0 %</encoder_util></utilization>
<clocks_event_reasons><sw_power_cap>Not Active</sw_power_cap></clocks_event_reasons>
<clocks_event_reasons_counters><sw_power_cap>32366342545 us</sw_power_cap>
<sw_therm_slowdown>976337 us</sw_therm_slowdown></clocks_event_reasons_counters>
</gpu></nvidia_smi_log>"""
        self.subject.command = lambda args: xml
        metrics = self.subject.gpu()
        self.assertIn('spark_gpu_temperature_limit_celsius{gpu="0",uuid="GPU-x"} 27.0', metrics)
        self.assertIn('field="temperature_limit_celsius"', metrics)
        self.assertIn('field="memory_temperature_celsius"', metrics)
        self.assertNotIn("spark_gpu_memory_temperature_celsius{", metrics)
        self.assertIn('spark_gpu_clock_throttle_seconds_total{gpu="0",reason="sw_power_cap",uuid="GPU-x"} 32366.342545', metrics)
        self.assertIn('spark_gpu_clock_throttle_active{gpu="0",reason="sw_power_cap",uuid="GPU-x"} 0', metrics)
        self.assertIn('spark_gpu_identity{detail="580.1",gpu="0",kind="driver",uuid="GPU-x"} 1', metrics)
        self.assertIn('spark_gpu_identity{detail="2E12-275-A1",gpu="0",kind="part",uuid="GPU-x"} 1', metrics)
        self.assertIn('spark_gpu_persistence_mode{gpu="0",uuid="GPU-x"} 1', metrics)
        self.assertIn('spark_gpu_c2c_mode{gpu="0",uuid="GPU-x"} 1', metrics)

    def test_engine_stats_rate_uses_observation_interval_and_detects_reset(self):
        self.responses[(8000, "/metrics")] = (
            b'vllm:prompt_tokens_total{model_name="m"} 1000\n'
            b'vllm:generation_tokens_total{model_name="m"} 500\n'
            b'vllm:num_requests_running{model_name="m"} 2\n'
            b'vllm:num_requests_waiting{model_name="m"} 1\n'
            b'vllm:kv_cache_usage_perc{model_name="m"} 0.4\n'
            b'vllm:time_to_first_token_seconds_sum{model_name="m"} 2.0\n'
            b'vllm:time_to_first_token_seconds_count{model_name="m"} 4\n',
            exporter.PROM_CONTENT_TYPE)
        subject = exporter.Exporter("100.100.1.2", self.root, self.config, (8000, 8000),
                                    command=self.command, fetch=self.fetch, enable_inference=True)
        subject.engine_counters[(("8000"), "vllm:generation_tokens_total")] = (400, exporter.time.time() - 2.0)
        subject.engine_counters[(("8000"), "vllm:prompt_tokens_total")] = (1000, exporter.time.time() - 2.0)
        metrics = subject.engine_stats()
        self.assertIn('spark_engine_tokens_total{direction="output",engine="vllm",engine_port="8000",model="m"} 500', metrics)
        rate = [float(line.split()[-1]) for line in metrics.splitlines()
                if line.startswith('spark_engine_tokens_per_second{direction="output"')][0]
        self.assertGreater(rate, 49.0)
        self.assertLess(rate, 51.0)
        # Unchanged counter must produce a measured zero rate, not a reset.
        self.assertIn('spark_engine_tokens_per_second{direction="input",engine="vllm",engine_port="8000",model="m"} 0', metrics)
        self.assertIn('spark_engine_requests_running{engine="vllm",engine_port="8000",model="m"} 2', metrics)
        self.assertIn('spark_engine_kv_cache_usage_ratio{engine="vllm",engine_port="8000",model="m"} 0.4', metrics)
        self.assertIn('spark_engine_time_to_first_token_seconds_mean{engine="vllm",engine_port="8000",model="m"} 0.5', metrics)
        self.assertIn('spark_engine_time_to_first_token_seconds_observations_total{engine="vllm",engine_port="8000",model="m"} 4', metrics)

        # A restarted engine's counter decreases: report reset, never a negative rate.
        self.responses[(8000, "/metrics")] = (
            b'vllm:generation_tokens_total{model_name="m"} 10\n'
            b'vllm:prompt_tokens_total{model_name="m"} 5\n',
            exporter.PROM_CONTENT_TYPE)
        metrics = subject.engine_stats()
        self.assertNotIn("spark_engine_tokens_per_second", metrics)
        self.assertIn('spark_engine_counter_reset_total{direction="output",engine="vllm",engine_port="8000",model="m"} 1', metrics)

    def test_engine_stats_absent_families_stay_absent(self):
        self.responses[(8000, "/metrics")] = (
            b'vllm:num_requests_running{model_name="m"} 1\n',
            exporter.PROM_CONTENT_TYPE)
        subject = exporter.Exporter("100.100.1.2", self.root, self.config, (8000, 8000),
                                    command=self.command, fetch=self.fetch, enable_inference=True)
        metrics = subject.engine_stats()
        self.assertIn('spark_engine_requests_running{engine="vllm",engine_port="8000",model="m"} 1', metrics)
        self.assertNotIn("spark_engine_tokens_total", metrics)
        self.assertNotIn("spark_engine_kv_cache_usage_ratio", metrics)
        self.assertNotIn("spark_engine_time_to_first_token_seconds_mean", metrics)

    def test_engine_stats_disabled_makes_no_probes(self):
        metrics = self.subject.engine_stats()
        self.assertEqual(metrics, "")
        self.assertEqual([call for call in self.calls if isinstance(call, tuple)], [])

    def test_tensorfold_speculation_and_cache_percentage_scale(self):
        self.responses[(8007, "/metrics")] = (
            b'tensorfold:prompt_tokens_total 57888701\n'
            b'tensorfold:generation_tokens_total 532722\n'
            b'tensorfold:kv_cache_usage_perc{pool="0"} 25\n'
            b'tensorfold:mtp_drafted_total 528103\n'
            b'tensorfold:mtp_accepted_total 369513\n'
            b'tensorfold:request_prefill_time_seconds_sum 3.0\n'
            b'tensorfold:request_prefill_time_seconds_count 2\n',
            exporter.PROM_CONTENT_TYPE)
        subject = exporter.Exporter("100.100.1.2", self.root, self.config, (8007, 8007),
                                    command=self.command, fetch=self.fetch, enable_inference=True)
        metrics = subject.engine_stats()
        self.assertIn('spark_engine_endpoint_up{engine="tensorfold",engine_port="8007"} 1', metrics)
        self.assertIn('spark_engine_mtp_draft_tokens_total{engine="tensorfold",engine_port="8007"} 528103', metrics)
        self.assertIn('spark_engine_kv_cache_usage_ratio{engine="tensorfold",engine_port="8007"} 0.25', metrics)
        self.assertIn('spark_engine_prefill_seconds_mean{engine="tensorfold",engine_port="8007"} 1.5', metrics)
        self.assertIn('spark_engine_speculation_acceptance_ratio{engine="tensorfold",engine_port="8007"}', metrics)

    def test_nic_error_counters_reported_for_physical_nics_only(self):
        self.put("sys/class/net/eth0/device/uevent", "DRIVER=fixture\nPCI_SLOT_NAME=0000:01:00.0\n")
        self.put("sys/class/net/eth0/speed", "200000\n")
        self.put("sys/class/net/eth0/statistics/rx_errors", "3\n")
        self.put("sys/class/net/eth0/statistics/tx_dropped", "7\n")
        self.put("sys/class/net/eth0/statistics/rx_crc_errors", "1\n")
        self.put("sys/class/net/lo/statistics/rx_errors", "99\n")
        metrics = self.subject.nics()
        self.assertIn('spark_nic_receive_errors_total{device="eth0"} 3', metrics)
        self.assertIn('spark_nic_transmit_dropped_total{device="eth0"} 7', metrics)
        self.assertIn('spark_nic_receive_crc_errors_total{device="eth0"} 1', metrics)
        self.assertNotIn('device="lo"', metrics)

    def test_supervisor_lane_ports_and_provenance(self):
        self.subject.enable_supervisor = True
        self.responses[(8090, "/v1/status")] = (json.dumps({
            "ready_aliases": ["m"], "lease_count": 0, "memory_budget_mb": 119000,
            "memory_used_mb": 112000, "auto_evict": False,
            "lanes": {"lane": {"alias": "m", "state": "ready", "port": 8007, "priority": 2,
                               "elapsed_seconds": 27063, "memory_mb": 112000, "leases": 0,
                               "unit_found": True, "display_name": "Model M",
                               "provenance": {"source": "org/model", "revision": "abc"},
                               "fences": ["node:box-a", "node:box-b"]}}},
        ).encode(), "application/json")
        metrics = self.subject.supervisor()
        self.assertIn('spark_supervisor_lane_port{lane="lane",model="m"} 8007', metrics)
        self.assertIn('spark_supervisor_lane_priority{lane="lane",model="m"} 2', metrics)
        self.assertIn('spark_supervisor_lane_elapsed_seconds{lane="lane",model="m"} 27063', metrics)
        self.assertIn('spark_supervisor_lane_fence_count{lane="lane",model="m"} 2', metrics)
        self.assertIn('spark_supervisor_lane_identity{detail="org/model",kind="source",lane="lane",model="m"} 1', metrics)
        self.assertIn('spark_supervisor_ready_alias_count 1', metrics)


class SingleFlightTests(unittest.TestCase):
    def test_overlapping_only_then_reprobes(self):
        flight = exporter.SingleFlight()
        entered, release = threading.Event(), threading.Event()
        calls, results = [], []
        def collect():
            calls.append(1)
            entered.set()
            self.assertTrue(release.wait(3))
            return len(calls)
        first = threading.Thread(target=lambda: results.append(flight.call("targets", collect)))
        first.start()
        self.assertTrue(entered.wait(3))
        # Observe a second waiter entering the same Future without timing-dependent sleeps.
        with flight.lock:
            pending = flight.pending["targets"]
        original = pending.result
        waiting = threading.Event()
        def result():
            waiting.set()
            return original()
        with patch.object(pending, "result", result):
            second = threading.Thread(target=lambda: results.append(flight.call("targets", collect)))
            second.start()
            self.assertTrue(waiting.wait(3))
            release.set()
            second.join(3)
            first.join(3)
        self.assertEqual(calls, [1])
        self.assertEqual(results, [1, 1])
        self.assertEqual(flight.call("targets", lambda: 2), 2)


if __name__ == "__main__":
    unittest.main()
