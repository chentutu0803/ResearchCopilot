from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.settings import get_settings


def _probe_cdp(endpoint: str) -> dict | None:
    probe_url = endpoint.rstrip("/") + "/json/version"
    try:
        with urlopen(probe_url, timeout=2) as response:
            return json.loads(response.read().decode("utf-8"))
    except (URLError, json.JSONDecodeError, TimeoutError):
        return None


def main() -> None:
    settings = get_settings()
    endpoint = settings.playwright_cdp_endpoint
    existing = _probe_cdp(endpoint)
    if existing is not None:
        print("浏览器会话已存在")
        print(f"CDP Endpoint: {endpoint}")
        print(f"Browser: {existing.get('Browser', '')}")
        return

    browser_path = Path(settings.playwright_executable_path)
    if not browser_path.exists():
        raise SystemExit(f"浏览器可执行文件不存在: {browser_path}")

    user_data_dir = settings.browser_user_data_path_resolved
    user_data_dir.mkdir(parents=True, exist_ok=True)
    log_path = settings.trace_path / "browser_session.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(browser_path),
        f"--remote-debugging-port={settings.browser_remote_debug_port}",
        f"--user-data-dir={user_data_dir}",
        "--no-sandbox",
        "--disable-setuid-sandbox",
        "--no-first-run",
        "--disable-dev-shm-usage",
        "--disable-background-networking",
        "--disable-sync",
        "--disable-default-apps",
        "--disable-popup-blocking",
        "--disable-features=TranslateUI",
        "--disable-features=AutomationControlled",
        "--start-maximized",
    ]
    headless = settings.playwright_headless or not bool(os.environ.get("DISPLAY"))
    if headless:
        cmd.append("--headless=new")

    with log_path.open("ab") as log_file:
        subprocess.Popen(  # noqa: S603
            cmd,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
        )

    deadline = time.time() + 20
    while time.time() < deadline:
        payload = _probe_cdp(endpoint)
        if payload is not None:
            print("浏览器会话已启动")
            print(f"CDP Endpoint: {endpoint}")
            print(f"Browser: {payload.get('Browser', '')}")
            print(f"User Data Dir: {user_data_dir}")
            print(f"Log: {log_path}")
            return
        time.sleep(1)

    raise SystemExit(
        "浏览器会话启动失败，未能在预期时间内打开 CDP 端口。"
    )


if __name__ == "__main__":
    main()
