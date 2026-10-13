"""Exercise the real dev-server mount with a fake model and temporary storage."""
from __future__ import annotations

import importlib.util
import os
import threading
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
import requests

import http_api
from test_lab import FakeProvider
from service import Service


@pytest.fixture
def server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, Service]:
    monkeypatch.delenv("AI_ORIGIN_TOKEN", raising=False)
    app = Service(tmp_path / "data", FakeProvider())
    monkeypatch.setattr(http_api, "_service", app)
    # By default this suite needs only this checkout. An explicit path also
    # exercises Kvasir's optional integration with exactly the same tests.
    integration = os.environ.get("KVASIR_DEV_SERVER")
    path = Path(integration) if integration else Path(__file__).with_name("dev.py")
    spec = importlib.util.spec_from_file_location("lab_test_dev_server", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(module.DevHTTPRequestHandler, directory=str(getattr(module, "KVASIR_ROOT", tmp_path))))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", app
    server.shutdown()
    server.server_close()
    app.queue.join()
    app.close()


def test_static_mount_api_and_existing_routes(server: tuple[str, Service]) -> None:
    base, _ = server
    response = requests.get(base + "/ai-origin/", timeout=5)
    assert response.ok and "AI-origin lab" in response.text
    assert requests.get(base + "/ai-origin/app.js", timeout=5).ok
    if os.environ.get("KVASIR_DEV_SERVER"):
        assert requests.get(base + "/exams", timeout=5).ok
    else:
        assert requests.get(base + "/", timeout=5).url.endswith("/ai-origin/")
        assert requests.get(base + "/src/ai-origin/service.py", timeout=5).status_code == 404
    assert requests.get(base + "/ai-origin/style.css", timeout=5).ok
    assert requests.get(base + "/ai-origin/../../../.git/config", timeout=5).status_code == 404
    assert requests.get(base + "/ai-origin/api/config", timeout=5).json()["repo_prompt"]["text"]
    assert requests.get(base + "/ai-origin/api/config", timeout=5).json()["graph_engine"] == "LangGraph"
    assert requests.get(base + "/ai-origin/api/models", timeout=5).json()["models"][0]["id"] == "test-model"
    assert requests.get(base + "/ai-origin/%2e%2e/ai-origin/../../AGENTS.md", timeout=5).status_code == 404


def test_write_guards_and_validation(server: tuple[str, Service]) -> None:
    base, _ = server
    url = base + "/ai-origin/api/samples"
    assert requests.post(url, json={}, timeout=5).status_code == 403
    headers = {"X-AI-Origin": "1", "Origin": "https://unrelated.example"}
    assert requests.post(url, json={}, headers=headers, timeout=5).status_code == 403
    response = requests.post(url, json={}, headers={"X-AI-Origin": "1"}, timeout=5)
    assert response.status_code == 400 and response.json()["details"]
    assert requests.get(base + "/ai-origin/api/config", headers={"Host": "hostile.example"}, timeout=5).status_code == 403


def test_remote_token(server: tuple[str, Service], monkeypatch: pytest.MonkeyPatch) -> None:
    base, _ = server
    monkeypatch.setenv("AI_ORIGIN_TOKEN", "test-access")
    assert requests.get(base + "/ai-origin/api/config", timeout=5).status_code == 403
    response = requests.get(base + "/ai-origin/api/config", headers={"Authorization": "Bearer test-access"}, timeout=5)
    assert response.ok


def test_dashboard_workflow(server: tuple[str, Service], tmp_path: Path) -> None:
    """Browser test: forms -> run -> result -> improvement -> candidate -> backtest.

    Chromium is optional for other contributors, but this test is run locally
    before delivery. No API traffic leaves the temporary test server.
    """
    playwright = pytest.importorskip("playwright.sync_api")
    from shutil import which
    executable = which("google-chrome") or which("chromium")
    if not executable:
        pytest.skip("Install Chrome/Chromium for browser verification")
    base, app = server
    errors = []
    with playwright.sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=executable, headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1440, "height": 1050})
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.goto(base + "/ai-origin/")
        page.wait_for_function("document.querySelector('#connection').textContent.includes('connected')")
        page.get_by_role("button", name="Add text", exact=True).click()
        form = page.locator("#sample-form")
        for field, value in {"title": "A multilingual example", "lang": "he", "manual_probability": "20",
                             "manual_low": "10", "manual_high": "30", "text": "התשובה שלי. This is a test <script>alert(1)</script>"}.items():
            form.locator(f'[name="{field}"]').fill(value)
        page.get_by_role("button", name="Save sample", exact=True).click()
        page.wait_for_selector("#sample-dialog", state="hidden")
        # Closing the modal precedes the asynchronous list refresh. Wait for the
        # saved row, rather than depending on browser/network scheduling.
        playwright.expect(page.locator("#samples-body")).to_contain_text("A multilingual example")
        page.locator('[data-tab="evaluate"]').click()
        page.locator('#strategy').select_option('manual')
        page.locator('#manual-selection input').check()
        page.get_by_role("button", name="Start evaluation", exact=True).click()
        page.wait_for_function("document.querySelector('#run-summary').textContent.includes('1 completed')")
        assert "25.0%" in page.locator("#results-body").inner_text()
        assert "20.0%" in page.locator("#results-body").inner_text()
        page.screenshot(path=str(tmp_path / "results-desktop.png"), full_page=True)
        with page.expect_download() as download:
            page.get_by_role("button", name="Export CSV", exact=True).click()
        assert Path(download.value.path()).read_text().startswith('text_id,text_link')
        app.provider.replies = [{"prompt": "Improved policy. Return the existing fields.", "rationale": "Test improvement."}]
        page.locator('#improve-form [name="name"]').fill("Reviewed candidate")
        page.get_by_role("button", name="Generate candidate", exact=True).click()
        page.get_by_role("button", name="Review candidate", exact=True).wait_for()
        page.get_by_role("button", name="Review candidate", exact=True).click()
        page.wait_for_function("document.querySelector('#candidate-text').value.includes('Improved policy')")
        assert "Test improvement" in page.locator("#candidate-info").inner_text()
        page.locator('#review-form [name="note"]').fill('Ready for test')
        page.get_by_role("button", name="Mark approved", exact=True).click()
        page.wait_for_function("document.querySelector('#candidate-info').textContent.includes('approved')")
        page.get_by_role("button", name="Use in evaluation", exact=True).click()
        run_id = app.store.list("runs")[0]["id"]
        page.locator('#reference-run').select_option(run_id)
        page.get_by_role("button", name="Start evaluation", exact=True).click()
        page.wait_for_function("document.querySelector('#run-summary').textContent.includes('Paired backtest: 1')")
        page.set_viewport_size({"width": 390, "height": 844})
        page.locator('[data-tab="samples"]').click()
        page.get_by_role("button", name="Add text", exact=True).click()
        page.screenshot(path=str(tmp_path / "sample-mobile.png"), full_page=True)
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        # Deliberate test examples stay separate from both Development and Holdout.
        for field, value in {"title": "Deliberate test", "text": "My test answer.",
                             "manual_probability": "20", "manual_low": "10", "manual_high": "30"}.items():
            form.locator(f'[name="{field}"]').fill(value)
        form.locator('[name="split"]').select_option("test")
        page.get_by_role("button", name="Save sample", exact=True).click()
        page.wait_for_selector("#sample-dialog", state="hidden")
        page.locator('#split-filter').select_option('test')
        playwright.expect(page.locator('#samples-body')).to_contain_text('Deliberate test')
        playwright.expect(page.locator('#samples-body')).not_to_contain_text('A multilingual example')
        page.locator('#samples-body [data-edit-sample]').click()
        playwright.expect(form.locator('[name="split"]')).to_have_value('test')
        page.get_by_role("button", name="Save sample", exact=True).click()
        page.wait_for_selector("#sample-dialog", state="hidden")
        page.locator('[data-tab="evaluate"]').click()
        page.locator('#reference-run').select_option('')
        page.locator('#run-split').select_option('test')
        page.locator('#strategy').select_option('all')
        page.get_by_role("button", name="Start evaluation", exact=True).click()
        playwright.expect(page.locator('#results-body')).to_contain_text('Deliberate test')
        page.wait_for_function("document.querySelector('#run-summary').textContent.includes('1 completed')")
        playwright.expect(page.locator('#improve-panel')).to_be_hidden()
        assert any(r['config']['split'] == 'test' and r['results'][0]['sample']['split'] == 'test'
                   for r in app.store.list('runs'))
        browser.close()
    assert errors == []
