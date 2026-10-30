"""Compare Python and curl HTTPS reachability without sending API keys or CVs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import http.client
import json
import math
import os
from pathlib import Path
import re
import shutil
import ssl
import subprocess
import sys
from time import perf_counter
import urllib.error
import urllib.request

from scripts.check_runtime import load_environment

ROOT_URL = "https://generativelanguage.googleapis.com/"
PROMPT = json.dumps({"contents": [{"parts": [{"text": "Reply with exactly MODEL_OK"}]}]}).encode()


def python_probe(url, method, timeout):
    request = urllib.request.Request(
        url, method=method, data=PROMPT if method == "POST" else None,
        headers={"Content-Type": "application/json"} if method == "POST" else {},
    )
    started = perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = {"http_response_received": True, "http_status": response.status}
    except urllib.error.HTTPError as exc:
        # Authentication errors still demonstrate HTTP reachability. Discard their bodies.
        result = {"http_response_received": True, "http_status": exc.code}
        exc.close()
    except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
        reason = getattr(exc, "reason", exc)
        result = {"http_response_received": False, "error_type": type(reason).__name__}
        if isinstance(getattr(reason, "errno", None), int):
            result["os_error"] = reason.errno
    return {**result, "elapsed_ms": round((perf_counter() - started) * 1000, 3)}


def curl_probe(url, method, timeout):
    executable = shutil.which("curl.exe" if os.name == "nt" else "curl")
    if not executable:
        return {"http_response_received": False, "error_type": "CurlNotInstalled"}
    command = [executable, "--silent", "--output", os.devnull,
               "--write-out", "%{http_code} %{time_connect} %{time_appconnect} %{time_starttransfer}",
               "--connect-timeout", str(min(timeout, 5)), "--max-time", str(timeout)]
    if method == "HEAD":
        command += ["--head"]
    else:
        command += ["--request", "POST", "--header", "Content-Type: application/json", "--data-binary", "@-"]
    command.append(url)
    started = perf_counter()
    try:
        process = subprocess.run(command, input=PROMPT if method == "POST" else None,
                                 capture_output=True, timeout=timeout + 3, check=False)
        # Do not log stderr, response content, proxy URLs or arbitrary exception messages.
        fields = process.stdout.decode("ascii").strip().split()
        code = int(fields[0])
        times = [float(value) for value in fields[1:]]
        if len(times) != 3 or not all(math.isfinite(value) and value >= 0 for value in times):
            raise ValueError("Unexpected timing result")
        if code != 0 and not 100 <= code <= 599:
            raise ValueError("Unexpected HTTP status")
        result = {"http_response_received": code != 0, "http_status": code or None,
                  "curl_exit_code": process.returncode,
                  **dict(zip(("tcp_connect_seconds", "tls_ready_seconds", "first_byte_seconds"), times))}
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError, UnicodeError) as exc:
        result = {"http_response_received": False, "error_type": type(exc).__name__}
    return {**result, "elapsed_ms": round((perf_counter() - started) * 1000, 3)}


def proxy_summary():
    try:
        proxies = urllib.request.getproxies()
        return {"http_proxy_detected": bool(proxies.get("http")),
                "https_proxy_detected": bool(proxies.get("https")),
                "all_proxy_detected": bool(proxies.get("all")),
                "target_bypasses_proxy": urllib.request.proxy_bypass("generativelanguage.googleapis.com"),
                "proxy_environment_variables_present": sorted(
                    name for name, value in os.environ.items()
                    if name.lower() in {"http_proxy", "https_proxy", "all_proxy", "no_proxy"} and value)}
    except OSError as exc:
        return {"error_type": type(exc).__name__}


def main():
    load_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=10, help="Per-operation timeout, 0 < seconds <= 30")
    parser.add_argument("--report", type=Path, default=Path("data/benchmarks/gemini-connection-check.json"))
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 30:
        parser.error("--timeout must be finite, positive and at most 30 seconds")
    model = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite").removeprefix("models/")
    if not re.fullmatch(r"gemini-[a-zA-Z0-9._-]+", model):
        parser.error("GEMINI_CHAT_MODEL is not a valid model identifier")
    post_url = ROOT_URL + "v1beta/models/" + model + ":generateContent"
    specs = [(client, method, url) for client in ("python", "curl")
             for method, url in (("HEAD", ROOT_URL), ("POST", post_url))]

    def run(spec):
        client, method, url = spec
        probe = python_probe if client == "python" else curl_probe
        return {"client": client, "method": method, **probe(url, method, args.timeout)}

    with ThreadPoolExecutor(max_workers=4) as pool:
        probes = list(pool.map(run, specs))
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "HTTPS reachability only. POST has no API key and should be rejected; not inference evidence.",
              "api_keys_transmitted": False, "cv_records_sent": 0,
              "python_version": sys.version.split()[0], "tls": ssl.OPENSSL_VERSION,
              "timeout_seconds": args.timeout, "proxy": proxy_summary(), "probes": probes}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    partial = args.report.with_suffix(".json.part")
    partial.write_text(json.dumps(report, indent=2), encoding="utf-8")
    partial.replace(args.report)
    print(json.dumps(report, indent=2))
    return 0 if all(probe["http_response_received"] for probe in probes) else 2


if __name__ == "__main__":
    raise SystemExit(main())
