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
from test_lab import FakeProvider, sample
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
        playwright.expect(page.locator('#run-model')).to_have_value('')
        playwright.expect(page.locator('#improve-model')).to_have_value('')
        # Model list refreshes always start with the requested default.
        page.route('**/ai-origin/api/models', lambda route: route.fulfill(json={"models": [
            {"id": "test-model", "efforts": ["auto"]},
            {"id": "gpt-6-luna", "efforts": ["auto", "high"]}]}))
        page.evaluate('loadModels()')
        playwright.expect(page.locator('#run-model')).to_have_value('gpt-6-luna')
        playwright.expect(page.locator('#improve-model')).to_have_value('gpt-6-luna')
        # Use the offline fake provider for the remaining workflow.
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
        page.get_by_label("Search sample text, title or ID").fill("THIS IS A TEST")
        with page.expect_response(lambda response: '/api/samples?q=' in response.url):
            page.wait_for_timeout(250)
        playwright.expect(page.locator("#samples-body")).to_contain_text("A multilingual example")
        page.locator('#sample-search').fill('not present in this sample')
        playwright.expect(page.locator("#samples-body")).not_to_contain_text("A multilingual example")
        page.locator('#sample-search').fill('')
        playwright.expect(page.locator("#samples-body")).to_contain_text("A multilingual example")
        page.locator('[data-tab="evaluate"]').click()
        page.locator('#run-model').select_option('test-model')
        page.locator('#strategy').select_option('manual')
        page.locator('#manual-selection input').check()
        page.get_by_role("button", name="Start evaluation", exact=True).click()
        page.wait_for_function("document.querySelector('#run-summary').textContent.includes('1 completed')")
        playwright.expect(page.locator('#run-summary')).to_contain_text('Development')
        playwright.expect(page.locator('#run-history')).to_contain_text('Development')
        assert "25.0%" in page.locator("#results-body").inner_text()
        assert "20.0%" in page.locator("#results-body").inner_text()
        # Exercise header clicks with numeric, negative and missing values.
        page.evaluate("""() => {
            window.savedSortRows = state.run.results;
            const base=state.run.results[0];
            state.run.results=[100,2,null,-5].map((difference,i)=>({...base,
                sample:{...base.sample,title:`Sort ${i}`,manual_probability:0},model_probability:difference}));
            renderResultRows();
        }""")
        difference = page.locator('[data-result-sort="difference"]')
        difference.click()
        assert page.locator('#results-body tr td:nth-child(5)').all_text_contents() == ['-5.0 pp','2.0 pp','100.0 pp','—']
        playwright.expect(difference.locator('..')).to_have_attribute('aria-sort','ascending')
        difference.focus()
        page.keyboard.press('Enter')
        assert page.locator('#results-body tr td:nth-child(5)').all_text_contents() == ['100.0 pp','2.0 pp','-5.0 pp','—']
        playwright.expect(difference.locator('..')).to_have_attribute('aria-sort','descending')
        page.evaluate('renderResultRows()')
        assert page.locator('#results-body tr td:nth-child(5)').all_text_contents()[-1] == '—'
        assert page.evaluate('state.run.results.map(r=>r.model_probability)') == [100,2,None,-5]
        page.evaluate('state.run.results=window.savedSortRows; renderResultRows()')
        playwright.expect(page.locator('#difference-histogram .histogram-bar[data-count="1"]')).to_have_count(1)
        playwright.expect(page.locator('#run-trends .metric-line')).to_have_count(3)
        playwright.expect(page.locator('#run-trends .metric-point')).to_have_count(3)
        assert page.evaluate("""() => {
            const row = (m,h,status='completed') => ({model_probability:m,sample:{manual_probability:h},status});
            const bins = RunCharts.histogramBins([row(0,100),row(100,0),row(20,20),row(null,20),row(50,20,'failed')]);
            return bins[0].count===1 && bins[10].count===1 && bins[19].count===1 && bins.reduce((n,b)=>n+b.count,0)===3;
        }""")
        # Chronology, dataset filters and missing metrics must not invent zeros.
        assert page.evaluate("""() => {
            const runs=[
                {id:'b',created_at:'2026-02-01',status:'completed',config:{split:'test'},metrics:{overall:{mae_pp:10,rmse_pp:12,bias_pp:-2}}},
                {id:'a',created_at:'2026-01-01',status:'completed_with_errors',config:{split:'holdout'},metrics:{overall:{mae_pp:null,rmse_pp:null,bias_pp:null}}},
                {id:'c',created_at:'2026-03-01',status:'running',config:{split:'test'}},
                {id:'d',created_at:'2026-04-01',status:'completed',config:{split:'all'}}];
            const target=document.createElement('div');
            RunCharts.trends(target,runs,'all',null);
            return RunCharts.trendRuns(runs,'all').map(r=>r.id).join(',')==='a,b,d'
                && RunCharts.trendRuns(runs,'test').length===1 && RunCharts.trendRuns(runs,'mixed')[0].id==='d'
                && target.querySelectorAll('.metric-point').length===3;
        }""")
        page.screenshot(path=str(tmp_path / "results-desktop.png"), full_page=True)
        with page.expect_download() as download:
            page.get_by_role("button", name="Export CSV", exact=True).click()
        assert Path(download.value.path()).read_text().startswith('text_id,text_link')
        app.provider.replies = [{"prompt": "Improved policy. Return the existing fields.", "rationale": "Test improvement."}]
        page.locator('#improve-model').select_option('test-model')
        page.locator('#improve-form [name="name"]').fill("Reviewed candidate")
        page.get_by_role("button", name="Generate candidate", exact=True).click()
        page.get_by_role("button", name="Open version", exact=True).wait_for()
        page.get_by_role("button", name="Open version", exact=True).click()
        page.wait_for_function("document.querySelector('#candidate-text').value.includes('Improved policy')")
        assert "Test improvement" in page.locator("#candidate-info").inner_text()
        assert page.get_by_role("button", name="Mark approved").count() == 0
        page.locator("#version-name").fill("Renamed version")
        page.get_by_role("button", name="Rename", exact=True).click()
        page.wait_for_function("document.querySelector('#candidate-select').selectedOptions[0].text === 'Renamed version'")
        page.get_by_role("button", name="Use in evaluation", exact=True).click()
        run_id = app.store.list("runs")[0]["id"]
        page.locator('#reference-run').select_option(run_id)
        page.get_by_role("button", name="Start evaluation", exact=True).click()
        page.wait_for_function("document.querySelector('#run-summary').textContent.includes('Paired backtest: 1')")
        playwright.expect(page.locator('#run-trends .metric-point')).to_have_count(6)
        page.locator('[data-tab="candidates"]').click()
        page.once("dialog", lambda dialog: dialog.accept())
        page.get_by_role("button", name="Remove version", exact=True).click()
        page.wait_for_function("document.querySelector('#candidate-select').value === 'repo'")
        playwright.expect(page.locator("#rename-form")).to_be_hidden()
        playwright.expect(page.locator("#candidate-select")).not_to_contain_text("Renamed version")
        playwright.expect(page.locator("#improvements-body")).to_contain_text("Version removed")
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
        playwright.expect(page.locator('#run-summary')).to_contain_text('Test')
        playwright.expect(page.locator('#improve-panel')).to_be_visible()
        page.locator('#trend-split').select_option('test')
        playwright.expect(page.locator('#run-trends .metric-point')).to_have_count(3)
        page.locator('#trend-split').select_option('holdout')
        playwright.expect(page.locator('#run-trends')).to_contain_text('Complete an evaluation')
        page.locator('#trend-split').select_option('all')
        playwright.expect(page.locator('#run-trends .metric-point')).to_have_count(9)
        page.locator('#run-trends circle.metric-point').first.focus()
        page.keyboard.press('Enter')
        playwright.expect(page.locator('#results-body')).to_contain_text('A multilingual example')
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.set_viewport_size({"width": 1440, "height": 1050})
        page.screenshot(path=str(tmp_path / "charts-desktop.png"), full_page=True)
        # Re-enter an older completed Test run and prepare a coder improvement.
        test_run = next(r for r in app.store.list('runs') if r['config']['split'] == 'test')
        page.locator('#run-history').select_option(test_run['id'])
        playwright.expect(page.locator('#improve-panel')).to_be_visible()
        playwright.expect(page.locator('#improve-scope')).to_have_value('all')
        page.locator('#improve-runner').select_option('codex')
        page.locator('#improve-form [name="name"]').fill('Coder from history')
        page.locator('#improve-form [name="guidance"]').fill('Prioritize exact evidence and lower false positives.')
        page.get_by_role('button', name='Prepare coder handoff', exact=True).click()
        playwright.expect(page.locator('#coder-handoff')).to_be_visible()
        playwright.expect(page.locator('#coder-prompt')).to_contain_text('instructions.md')
        job = next(j for j in app.store.list('improvements') if j['config'].get('runner') == 'codex')
        assert job['config']['guidance'] == 'Prioritize exact evidence and lower false positives.'
        playwright.expect(page.locator('#run-prompt')).to_contain_text('Coder from history')
        page.get_by_role('button', name='Load coder output', exact=True).click()
        playwright.expect(page.locator('#coder-status')).to_contain_text('still empty')
        Path(job['output_path']).write_text('Coder-authored prompt ready for backtesting.')
        page.get_by_role('button', name='Load coder output', exact=True).click()
        playwright.expect(page.locator('#candidate-text')).to_have_value('Coder-authored prompt ready for backtesting.')
        # Return again, without a reload, to make a fresh handoff with new guidance.
        page.locator('[data-tab="results"]').click()
        page.locator('#improve-runner').select_option('claude_code')
        page.locator('#improve-form [name="name"]').fill('Claude follow-up')
        page.locator('#improve-form [name="guidance"]').fill('Different guidance for this attempt.')
        page.get_by_role('button', name='Prepare coder handoff', exact=True).click()
        playwright.expect(page.locator('#coder-paths')).not_to_contain_text(job['candidate_id'])
        page.screenshot(path=str(tmp_path / 'coder-handoff.png'), full_page=True)
        assert any(r['config']['split'] == 'test' and r['results'][0]['sample']['split'] == 'test'
                   for r in app.store.list('runs'))
        browser.close()
    assert errors == []


def test_sample_search_scans_current_text_files(server: tuple[str, Service]) -> None:
    base, app = server
    first = app.save_sample(sample(title="First", text="A distinctive sentence. שלום עולם"))
    second = app.save_sample(sample(title="Second", text="A different answer."))
    def search(query):
        response = requests.get(base + "/ai-origin/api/samples", params={"q": query}, timeout=5)
        assert response.ok
        return response.json()
    assert [s["id"] for s in search(" DISTINCTIVE SENTENCE ")] == [first["id"]]
    assert [s["id"] for s in search("שלום")] == [first["id"]]
    assert [s["id"] for s in search("SECOND")] == [second["id"]]
    assert [s["id"] for s in search(first["id"])] == [first["id"]]
    assert "text" not in search("distinctive")[0]
    assert search("no such phrase") == []
    app.save_sample(sample(title="First", text="Updated wording."), first["id"], first["revision"])
    assert search("distinctive") == []
    assert [s["id"] for s in search("updated wording")] == [first["id"]]
    assert len(search("")) == 2



def test_one_text_settings_validation_and_no_storage(server: tuple[str, Service]) -> None:
    import json
    from schemas import CandidateInput
    from provider import ProviderError

    base, app = server
    candidate = app.save_candidate(CandidateInput(name="One-off policy", prompt="Custom evaluation policy."))
    before = {p.relative_to(app.store.root): p.read_bytes() for p in app.store.root.rglob("*") if p.is_file()}
    url = base + "/ai-origin/api/evaluate-one"
    headers = {"X-AI-Origin": "1"}
    data = {"text": "שלום <script>not code</script>", "lang": "he", "model": "test-model",
            "prompt_id": candidate["id"], "effort": "high", "max_output_tokens": 1024,
            "materials_mode": "use", "materials": "Course sources.", "assignment": "Explain your answer."}
    assert requests.post(url, json=data, timeout=5).status_code == 403
    response = requests.post(url, json=data, headers=headers, timeout=5)
    assert response.ok
    result = response.json()
    assert result["model_probability"] == 25
    assert result["explanation"] == "A quoted observation."
    assert result["raw_response"]["model"] == "test-model-snapshot"
    assert result["raw_response"]["usage"]["input_tokens"] == 100
    request = app.provider.requests[-1]
    assert request["model"] == "test-model" and request["reasoning"] == {"effort": "high"}
    assert request["max_output_tokens"] == 1024 and request["store"] is False
    assert request["instructions"].startswith("Custom evaluation policy.")
    assert "explanation language: he" in request["instructions"]
    assert json.loads(request["input"][0]["content"]) == {
        "lang": "he", "student_text": data["text"], "materials": data["materials"], "assignment": data["assignment"]}
    calls = len(app.provider.requests)
    for change in ({"text": "   "}, {"model": "unavailable"}, {"materials": ""},
                   {"max_output_tokens": 0}, {"manual_probability": 50}, {"sample_ids": ["saved"]}):
        assert requests.post(url, json={**data, **change}, headers=headers, timeout=5).status_code == 400
    assert len(app.provider.requests) == calls
    response = requests.post(url, json={"text": "Only this text.", "lang": None, "model": "test-model"},
                             headers=headers, timeout=5)
    assert response.ok
    request = app.provider.requests[-1]
    assert request["instructions"].startswith(app.prompt("repo")["text"])
    assert "explanation language: en" in request["instructions"]
    assert "reasoning" not in request
    # A refusal or failed paid call must never become a zero score or be retried.
    for reply in ({"status": "incomplete", "output": []}, ProviderError("Offline provider failure")):
        app.provider.replies = [reply]
        calls = len(app.provider.requests)
        response = requests.post(url, json=data, headers=headers, timeout=5)
        assert response.status_code == 400 and "model_probability" not in response.json()
        assert len(app.provider.requests) == calls + 1
    after = {p.relative_to(app.store.root): p.read_bytes() for p in app.store.root.rglob("*") if p.is_file()}
    assert after == before
    assert app.store.list("samples") == app.store.list("runs") == app.store.list("improvements") == []


def test_one_text_modal(server: tuple[str, Service], tmp_path: Path) -> None:
    from shutil import which
    from schemas import CandidateInput
    from provider import ProviderError
    playwright = pytest.importorskip("playwright.sync_api")
    executable = which("google-chrome") or which("chromium")
    if not executable:
        pytest.skip("Install Chrome/Chromium for browser verification")
    base, app = server
    candidate = app.save_candidate(CandidateInput(name="Quick check prompt", prompt="A candidate policy."))
    app.provider.replies = [{"ai_origin_probability": .73, "ai_origin_explanation": "Evidence. <script>alert(1)</script>"}]
    entered, release = threading.Event(), threading.Event()
    def pause_call():
        entered.set()
        assert release.wait(10)
    app.provider.on_call = pause_call
    errors = []
    with playwright.sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=executable, headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1440, "height": 1050})
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.goto(base + "/ai-origin/")
        page.wait_for_function("document.querySelector('#connection').textContent.includes('connected')")
        page.locator('[data-tab="evaluate"]').click()
        page.locator('#run-model').select_option('test-model')
        page.locator('#run-effort').select_option('high')
        page.locator('#run-prompt').select_option(candidate['id'])
        page.locator('#run-form [name="max_output_tokens"]').fill('1024')
        page.locator('#strategy').select_option('manual')  # No saved samples are needed.
        page.get_by_role('button', name='Evaluate one text', exact=True).click()
        form = page.locator('#one-text-form')
        playwright.expect(page.locator('#one-text-dialog')).to_be_visible()
        playwright.expect(page.locator('#one-text-settings')).to_contain_text('Quick check prompt · test-model · effort: high · output limit: 1024')
        page.locator('#submit-one-text').click()
        assert not app.provider.requests  # Empty textarea fails browser validation.
        form.locator('[name="text"]').fill('My one-off answer.')
        form.locator('[name="lang"]').fill('he')
        page.locator('#one-text-materials-mode').select_option('use')
        page.locator('#submit-one-text').click()
        assert not app.provider.requests  # Materials required only in use mode.
        form.locator('[name="materials"]').fill('Course context.')
        page.locator('#submit-one-text').click()
        playwright.expect(page.locator('#submit-one-text')).to_be_disabled()
        assert entered.wait(5)
        page.locator('#close-one-text').click()
        page.locator('#evaluate-one').click()
        playwright.expect(page.locator('#submit-one-text')).to_be_disabled()
        release.set()
        playwright.expect(page.locator('#one-text-score')).to_have_text('73.0%')
        playwright.expect(page.locator('#one-text-explanation')).to_have_text('Evidence. <script>alert(1)</script>')
        assert page.locator('#one-text-result script').count() == 0
        playwright.expect(page.locator('#one-text-json')).to_be_hidden()
        page.get_by_text('Full JSON response', exact=True).click()
        playwright.expect(page.locator('#one-text-json')).to_contain_text('test-model-snapshot')
        assert len(app.provider.requests) == 1
        request = app.provider.requests[0]
        assert request['instructions'].startswith('A candidate policy.')
        assert request['reasoning'] == {'effort': 'high'} and request['max_output_tokens'] == 1024
        page.screenshot(path=str(tmp_path / 'one-text-desktop.png'), full_page=True)
        page.locator('#close-one-text').click()
        page.locator('#evaluate-one').click()
        playwright.expect(page.locator('#one-text-score')).to_have_text('73.0%')
        page.set_viewport_size({'width': 390, 'height': 844})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        assert page.locator('#one-text-dialog').evaluate('(el) => el.scrollWidth <= el.clientWidth')
        page.screenshot(path=str(tmp_path / 'one-text-mobile.png'), full_page=True)
        app.provider.on_call = None
        app.provider.replies = [ProviderError('Test provider failure')]
        page.locator('#submit-one-text').click()
        playwright.expect(page.locator('#one-text-status')).to_contain_text('Test provider failure')
        playwright.expect(page.locator('#one-text-result')).to_be_hidden()
        playwright.expect(page.locator('#submit-one-text')).to_be_enabled()
        assert len(app.provider.requests) == 2
        assert app.store.list('samples') == app.store.list('runs') == app.store.list('improvements') == []
        browser.close()
    assert errors == []
