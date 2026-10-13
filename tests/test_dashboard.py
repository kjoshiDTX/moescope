"""Serve a fixture runs folder and check every dashboard view renders, offline, without console errors."""

import json
import socket
import threading
import time
import urllib.request

import pytest
import uvicorn

from moescope.server.api import STATIC, create_app

from .fixtures.sample_runs import make_sample_runs

VIEWS = {
    "utilization": "heatmap",
    "coverage": "coverage-columns",
    "coactivation": "coactivation-matrix",
    "compare": "compare-delta",
    "cache": "cache-chart",
    "prune": "prune-chart",
    "training": "train-ratio",
}


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    root = make_sample_runs(tmp_path_factory.mktemp("runs"))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(create_app(root), host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    thread.join(5)


def get(url):
    with urllib.request.urlopen(url) as r:
        return json.load(r)


def test_api_endpoints(server):
    runs = get(f"{server}/api/runs")
    assert {r["name"] for r in runs} == {"sample-code", "sample-chat"}
    report = get(f"{server}/api/runs/sample-code/report")
    assert len(report["utilization"]) == report["manifest"]["num_layers"]
    co = get(f"{server}/api/runs/sample-code/coactivation?layer=1")
    assert len(co["counts"]) == report["manifest"]["num_experts"]
    assert get(f"{server}/api/compare?a=sample-code&b=sample-chat")["mean_js_distance"] > 0
    assert get(f"{server}/api/prune")[0]["source_run"] == "sample-code"
    assert get(f"{server}/api/train")[0]["steps"]


@pytest.mark.parametrize("path", ["/api/runs/../../etc/report", "/api/runs/nope/report", "/api/runs/sample-code/coactivation?layer=99"])
def test_api_rejects_bad_paths(server, path):
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(server + path)
    assert e.value.code in (400, 404)


@pytest.mark.skipif(not (STATIC / "index.html").exists(), reason="dashboard not built (npm run build in web/)")
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_every_view_renders_offline(server, theme):
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(color_scheme=theme, viewport={"width": 1280, "height": 900})
        errors, external = [], []
        page.on("console", lambda m: m.type == "error" and errors.append(m.text))
        page.on("pageerror", lambda e: errors.append(str(e)))

        def guard(route):
            if route.request.url.startswith(server):
                route.continue_()
            else:
                external.append(route.request.url)
                route.abort()

        page.route("**/*", guard)
        for view, testid in VIEWS.items():
            page.goto(f"{server}/#view={view}&run=sample-code")
            page.get_by_test_id(testid).locator("svg").first.wait_for(timeout=10_000)
            # Hovering the chart must produce a tooltip without errors.
            box = page.get_by_test_id(testid).locator("svg").first.bounding_box()
            page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        page.goto(f"{server}/#view=training")
        page.get_by_test_id("collapse-warning").wait_for()
        browser.close()
    assert not errors, errors
    assert not external, f"dashboard fetched external resources: {external}"
