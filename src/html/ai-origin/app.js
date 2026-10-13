'use strict';

const $ = id => document.getElementById(id);
const state = {samples: [], runs: [], candidates: [], improvements: [], models: [], run: null, candidate: 'repo', sample: null, snapshot: false, tab: 'samples'};
const active = status => ['queued', 'running', 'cancelling'].includes(status);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = (value, suffix = '') => value == null ? '—' : `${Number(value).toFixed(1)}${suffix}`;
const rate = value => value == null ? '—' : fmt(value * 100, '%');
const badge = status => `<span class="badge ${esc(status)}">${esc(status.replaceAll('_', ' '))}</span>`;
const when = value => value ? new Date(value).toLocaleString() : '—';
const link = (id, run = '') => `/ai-origin/?sample=${encodeURIComponent(id)}${run ? '&run=' + encodeURIComponent(run) : ''}`;
const datasetLabel = split => ({development:'Development', holdout:'Holdout', test:'Test', all:'All datasets'}[split] || '');
function runLabel(run) {
  const dataset = datasetLabel(run.config?.split);
  if (!dataset || String(run.name || '').endsWith(dataset)) return run.name;
  return `${run.name} · ${dataset}`;
}

function notice(message, error = false) {
  $('notice').textContent = message; $('notice').hidden = !message; $('notice').className = error ? 'error' : '';
  document.querySelectorAll('.dialog-notice').forEach(el => el.remove());
  const dialog = document.querySelector('dialog[open]');
  if (message && dialog) {
    const alert = document.createElement('p'); alert.className = 'dialog-notice positive'; alert.setAttribute('role','alert'); alert.textContent = message;
    dialog.querySelector('.section-head').after(alert); alert.scrollIntoView({block:'nearest'});
  }
}

async function api(path, body, raw = false) {
  const headers = {'X-AI-Origin': '1'};
  const token = sessionStorage.getItem('ai-origin-token');
  if (token) headers.Authorization = `Bearer ${token}`;
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  const response = await fetch('/ai-origin/api/' + path, {method: body === undefined ? 'GET' : 'POST', headers, body: body === undefined ? undefined : JSON.stringify(body)});
  if (!response.ok) {
    const error = await response.json().catch(() => ({error: 'Server did not return JSON.'}));
    throw new Error(error.error + (error.details ? '\n' + error.details.map(e => `${e.field}: ${e.message}`).join('\n') : ''));
  }
  return raw ? response.text() : response.json();
}

async function action(fn, button) {
  if (button) button.disabled = true;
  notice('');
  try { await fn(); } catch (error) { notice(error.message, true); }
  finally { if (button) button.disabled = false; }
}

function switchTab(tab) {
  state.tab = tab;
  document.querySelectorAll('.view').forEach(view => view.hidden = view.id !== `${tab}-view`);
  document.querySelectorAll('[data-tab]').forEach(button => button.classList.toggle('active', button.dataset.tab === tab));
}

function fillSelect(id, options, fallback = '') {
  const el = $(id), old = el.value;
  el.innerHTML = options.map(([value, title]) => `<option value="${esc(value)}">${esc(title)}</option>`).join('');
  if (options.some(([value]) => value === old)) el.value = old;
  else if (options.some(([value]) => value === fallback)) el.value = fallback;
}

async function loadLists() {
  [state.samples, state.runs, state.candidates, state.improvements] = await Promise.all(['samples', 'runs', 'candidates', 'improvements'].map(path => api(path)));
  $('sample-count').textContent = state.samples.length;
  fillSelect('language-filter', [['', 'All languages'], ...[...new Set(state.samples.map(s => s.lang))].sort().map(v => [v,v])]);
  const prompts = [['repo', 'Repository prompt'], ...state.candidates.map(c => [c.id, c.name])];
  fillSelect('run-prompt', prompts, 'repo'); fillSelect('candidate-select', prompts, 'repo');
  fillSelect('run-history', [['', 'Choose a run'], ...state.runs.map(r => [r.id, `${when(r.created_at)} · ${runLabel(r)} · ${r.status}`])]);
  fillSelect('reference-run', [['', 'New selection'], ...state.runs.filter(r => !active(r.status)).map(r => [r.id, `${runLabel(r)} · ${r.id}`])]);
  renderSelection(); renderImprovements(); renderTrends(); await searchSamples();
}

let sampleSearchTimer, sampleSearchVersion = 0, matchingSampleIds = null;

async function searchSamples() {
  const version = ++sampleSearchVersion, query = $('sample-search').value.trim();
  if (!query) { matchingSampleIds = null; renderSamples(); return; }
  const matches = await api('samples?q=' + encodeURIComponent(query));
  // A slower previous request must not overwrite the latest search or a clear.
  if (version !== sampleSearchVersion || query !== $('sample-search').value.trim()) return;
  matchingSampleIds = new Set(matches.map(s => s.id));
  renderSamples();
}

function renderSamples() {
  const samples = state.samples.filter(s => (matchingSampleIds === null || matchingSampleIds.has(s.id)) && (!$('language-filter').value || s.lang === $('language-filter').value) && (!$('split-filter').value || s.split === $('split-filter').value));
  $('samples-body').innerHTML = samples.map(s => `<tr><td><a href="${link(s.id)}" target="_blank" rel="noopener">${esc(s.title)}</a><small>${esc(s.id)}</small></td><td>${esc(s.lang)}</td><td>${fmt(s.manual_probability,'%')}<small>${fmt(s.manual_low)}–${fmt(s.manual_high)}%</small></td><td>${esc(s.origin)}</td><td>${esc(s.materials_mode)}</td><td>${esc(s.split)}</td><td><button class="secondary" data-edit-sample="${s.id}">Edit</button></td></tr>`).join('') || '<tr><td colspan="7" class="empty">No samples yet. Add a text and your manual assessment.</td></tr>';
}

function renderSelection() {
  const checked = new Set([...$('manual-selection').querySelectorAll('input:checked')].map(el => el.value));
  const split = $('run-split').value, strategy = $('strategy').value, reference = $('reference-run').value;
  const samples = state.samples.filter(s => split === 'all' || s.split === split);
  $('manual-selection').innerHTML = samples.map(s => `<label><input type="checkbox" value="${s.id}" ${checked.has(s.id) ? 'checked' : ''}><span>${esc(s.title)} · ${esc(s.lang)} · ${fmt(s.manual_probability,'%')}</span></label>`).join('') || '<p>No eligible samples.</p>';
  $('manual-selection').hidden = strategy !== 'manual' || !!reference;
  $('percent-field').hidden = $('seed-field').hidden = strategy !== 'balanced' || !!reference;
  $('selection-help').hidden = strategy !== 'balanced' || !!reference;
  $('run-split').disabled = $('strategy').disabled = !!reference;
  $('selected-count').textContent = reference ? 'Uses the exact saved samples from the reference run.' : `${samples.length} eligible samples`;
}

function setEfforts(prefix) {
  const model = state.models.find(m => m.id === $(`${prefix}-model`).value);
  fillSelect(`${prefix}-effort`, (model?.efforts || ['auto']).map(v => [v, v === 'auto' ? 'Auto (model default)' : v]), 'auto');
}

async function loadModels() {
  const response = await api('models'); state.models = response.models;
  const defaultModel = 'gpt-6-luna';
  const available = state.models.some(m => m.id === defaultModel);
  const options = state.models.map(m => [m.id,m.id]);
  if (!available) options.unshift(['', `${defaultModel} unavailable — choose a model`]);
  for (const prefix of ['run', 'improve']) {
    fillSelect(`${prefix}-model`, options);
    $(`${prefix}-model`).value = available ? defaultModel : '';
    setEfforts(prefix);
  }
  $('connection').textContent = `OpenAI connected · ${state.models.length} text models`;
}

function setMaterials() {
  const use = $('materials-mode').value === 'use';
  $('materials-field').hidden = !use;
  $('sample-form').elements.materials.required = use;
}

function editSample(sample = null, snapshot = false) {
  state.sample = sample; state.snapshot = snapshot;
  const form = $('sample-form'); form.reset();
  for (const field of form.elements) {
    field.disabled = snapshot;
    if (field.name && sample && sample[field.name] != null) field.value = sample[field.name];
  }
  $('sample-heading').textContent = snapshot ? 'Run sample snapshot' : sample ? 'Edit sample' : 'Add a sample';
  $('snapshot-note').hidden = !snapshot; $('save-sample').hidden = snapshot;
  $('sample-id').textContent = sample ? `${sample.id} · revision ${sample.revision}` : '';
  setMaterials(); $('sample-dialog').showModal();
}

function readForm(form, numeric = []) {
  const data = Object.fromEntries(new FormData(form));
  for (const key of numeric) if (key in data) data[key] = Number(data[key]);
  return data;
}

async function showRun(id) {
  if (!id) { state.run = null; return; }
  state.run = await api('runs/' + id); $('run-history').value = id;
  renderRun();
}

function renderTrends() {
  // The selected detail may be fresher than the lightweight history poll.
  const runs = state.runs.map(r => r.id === state.run?.id ? state.run : r);
  RunCharts.trends($('run-trends'), runs, $('trend-split').value, state.run?.id);
}

function renderRun() {
  const run = state.run;
  if (!run) return;
  renderTrends(); RunCharts.histogram($('difference-histogram'), run);
  const m = run.metrics.overall, k = m.known_origin;
  $('run-summary').innerHTML = `<strong>${esc(runLabel(run))}</strong> ${badge(run.status)}<p>${m.completed} completed · ${m.failed} failed · ${m.pending} unfinished · ${esc(run.config.model)} / ${esc(run.config.effort)}</p><small>${esc(run.id)} · ${esc(run.prompt.name)} · SHA ${esc(run.prompt.sha256.slice(0,12))} · ${when(run.created_at)}</small>${run.error ? `<p class="positive">${esc(run.error)}</p>` : ''}${run.comparison ? `<p>Paired backtest: ${run.comparison.paired_samples} samples · MAE change ${fmt(run.comparison.mae_change_pp,' pp')} (negative is better) · ${run.comparison.improved} improved / ${run.comparison.regressed} regressed.</p>` : ''}`;
  const items = [['Mean absolute error',fmt(m.mae_pp,' pp')],['Root mean square error',fmt(m.rmse_pp,' pp')],['Signed bias',fmt(m.bias_pp,' pp')],['Inside manual range',rate(m.within_range)],['Distance outside range',fmt(m.range_distance_pp,' pp')],['Known-origin accuracy',rate(k.accuracy)],['Human false-positive rate',rate(k.false_positive_rate)],['AI recall',rate(k.recall)]];
  $('metrics').innerHTML = items.map(([name,value]) => `<div class="metric"><b>${value}</b><span>${name}</span></div>`).join('');
  $('run-summary').innerHTML += `<p class="help">Manual agreement uses ${m.completed} samples. Known-origin metrics use ${k.count} confirmed human/AI samples at a ${k.threshold}% threshold: TP ${k.tp}, TN ${k.tn}, FP ${k.fp}, FN ${k.fn}. Precision ${rate(k.precision)} · Brier ${k.brier == null ? '—' : k.brier.toFixed(3)}. These estimates are uncalibrated; this sample collection is not a population accuracy claim.</p>`;
  $('metric-details').hidden = false;
  $('grouped-metrics').innerHTML = `<table><thead><tr><th>Group</th><th>Value</th><th>Completed / total</th><th>MAE</th><th>Within range</th><th>Human FPR</th></tr></thead><tbody>${['lang','materials_mode','split'].flatMap(field => Object.entries(run.metrics[field]).map(([name,v]) => `<tr><td>${esc(field)}</td><td>${esc(name)}</td><td>${v.completed} / ${v.total}</td><td>${fmt(v.mae_pp,' pp')}</td><td>${rate(v.within_range)}</td><td>${rate(v.known_origin.false_positive_rate)}</td></tr>`)).join('')}</tbody></table>`;
  renderResultRows();
  $('cancel-run').hidden = !active(run.status);
  $('improve-panel').hidden = active(run.status) || !run.results.some(r => r.status === 'completed' && r.sample.split === 'development');
}

// Table ordering is a view preference: never reorder the stored run snapshots.
let resultSort = {key: null, direction: 1};
const resultCollator = new Intl.Collator(undefined, {numeric: true, sensitivity: 'base'});
function sortedResults(rows) {
  if (!resultSort.key) return rows;
  const value = row => {
    const sample = row.sample;
    switch (resultSort.key) {
      case 'text': return sample.title || sample.id;
      case 'lang': return sample.lang;
      case 'manual': return sample.manual_probability;
      case 'model': return row.model_probability;
      case 'difference': return row.model_probability == null ? null : row.model_probability - sample.manual_probability;
      case 'result': return `${row.status} ${row.explanation || row.error || ''}`;
    }
  };
  return [...rows].sort((a,b) => {
    const av = value(a), bv = value(b);
    // Unfinished/failed predictions stay last in either direction.
    const absent = v => v == null || (typeof v === 'number' && !Number.isFinite(v));
    if (absent(av) || absent(bv)) return Number(absent(av)) - Number(absent(bv));
    return resultSort.direction * (typeof av === 'number' ? av-bv : resultCollator.compare(av,bv));
  });
}

function renderResultRows() {
  const run = state.run;
  if (!run) return;
  document.querySelectorAll('[data-result-sort]').forEach(button => {
    const selected = button.dataset.resultSort === resultSort.key;
    button.closest('th').setAttribute('aria-sort', selected ? (resultSort.direction === 1 ? 'ascending' : 'descending') : 'none');
    button.querySelector('.sort-direction').textContent = selected ? (resultSort.direction === 1 ? '↑' : '↓') : '↕';
  });
  $('results-body').innerHTML = sortedResults(run.results).map(r => {
    const s = r.sample, difference = r.model_probability == null ? null : r.model_probability - s.manual_probability;
    return `<tr><td><a href="${link(s.id,run.id)}" target="_blank" rel="noopener">${esc(s.title)}</a><small>${esc(s.id)} · r${s.revision}</small></td><td>${esc(s.lang)}</td><td>${fmt(s.manual_probability,'%')}<small>${fmt(s.manual_low)}–${fmt(s.manual_high)}% · ${esc(s.origin)}</small></td><td>${fmt(r.model_probability,'%')}</td><td>${fmt(difference,' pp')}</td><td class="explanation">${badge(r.status)}${r.explanation ? `<details><summary>Read explanation</summary><p dir="auto">${esc(r.explanation)}</p></details>` : ''}${r.error ? `<p>${esc(r.error)}</p>` : ''}</td></tr>`;
  }).join('');
}

async function showCandidate(id) {
  state.candidate = id; $('candidate-select').value = id;
  const form = $('candidate-form'); form.reset();
  const saved = id !== 'repo';
  $('rename-form').hidden = !saved;
  if (!saved) {
    const prompt = await api('prompt'); $('candidate-text').value = prompt.text;
    $('candidate-info').textContent = `Repository prompt · SHA ${prompt.sha256.slice(0,12)}. Rename and remove apply to saved versions. Edits here are saved as a new version.`;
    $('candidate-diff').hidden = true;
  } else {
    const candidate = await api('candidates/' + id);
    $('candidate-text').value = candidate.prompt;
    $('version-name').value = candidate.name;
    $('candidate-info').innerHTML = `<p>${esc(candidate.rationale || 'No notes on this version.')}</p><small>${esc(candidate.id)} · ${when(candidate.created_at)} · SHA ${esc(candidate.sha256.slice(0,12))}</small>`;
    $('diff-text').textContent = candidate.diff || 'No changes from parent.';
    $('candidate-diff').hidden = false;
  }
}

function renderImprovements() {
  $('improvements-body').innerHTML = state.improvements.map(j => {
    const version = !j.candidate_id ? esc(j.error || 'The version will appear when complete.')
      : state.candidates.some(c => c.id === j.candidate_id) ? `<button class="secondary" data-open-candidate="${j.candidate_id}">Open version</button>`
      : 'Version removed';
    return `<tr><td>${when(j.created_at)}<small>${esc(j.id)}</small></td><td><button class="secondary" data-open-run="${j.config.run_id}">${esc(j.config.run_id)}</button></td><td>${esc(j.config.model)}<small>${esc(j.config.effort)}</small></td><td>${badge(j.status)}</td><td>${version}<details><summary data-job="${j.id}">View step details</summary><pre></pre></details></td></tr>`;
  }).join('') || '<tr><td colspan="5" class="empty">No improvement steps yet.</td></tr>';
}

async function download(path, filename) {
  const text = await api(path, undefined, true), url = URL.createObjectURL(new Blob([text], {type:'text/plain;charset=utf-8'}));
  const a = document.createElement('a'); a.href = url; a.download = filename; a.click(); URL.revokeObjectURL(url);
}

document.querySelectorAll('[data-tab]').forEach(button => button.addEventListener('click', () => switchTab(button.dataset.tab)));
['language-filter','split-filter'].forEach(id => $(id).addEventListener('input', renderSamples));
$('sample-search').addEventListener('input', () => {
  ++sampleSearchVersion; clearTimeout(sampleSearchTimer);
  if (!$('sample-search').value.trim()) { matchingSampleIds = null; renderSamples(); return; }
  sampleSearchTimer = setTimeout(() => action(searchSamples), 200);
});
['run-split','strategy','reference-run'].forEach(id => $(id).addEventListener('change', renderSelection));
['run','improve'].forEach(prefix => $(`${prefix}-model`).addEventListener('change', () => setEfforts(prefix)));
$('materials-mode').addEventListener('change', setMaterials);
$('new-sample').onclick = () => editSample();
$('close-sample').onclick = () => $('sample-dialog').close();
$('close-text').onclick = () => $('text-dialog').close();
$('refresh').onclick = () => action(loadLists, $('refresh'));
$('refresh-results').onclick = () => action(async () => { await loadLists(); if (state.run) await showRun(state.run.id); });
$('trend-split').onchange = renderTrends;
$('run-trends').onclick = event => { const id = event.target.closest('[data-run]')?.dataset.run; if (id) action(() => showRun(id)); };
$('run-trends').onkeydown = event => {
  if (event.target.matches('circle[data-run]') && ['Enter',' '].includes(event.key)) { event.preventDefault(); action(() => showRun(event.target.dataset.run)); }
};
$('results-table').querySelector('thead').onclick = event => {
  const key = event.target.closest('[data-result-sort]')?.dataset.resultSort;
  if (!key) return;
  resultSort = {key, direction: resultSort.key === key ? -resultSort.direction : 1};
  renderResultRows();
};
$('run-history').onchange = () => action(() => showRun($('run-history').value));
$('candidate-select').onchange = () => action(() => showCandidate($('candidate-select').value));
$('new-candidate').onclick = () => action(async () => { await showCandidate('repo'); $('candidate-text').focus(); });
$('samples-body').onclick = event => { const id = event.target.closest('[data-edit-sample]')?.dataset.editSample; if (id) action(async () => editSample(await api('samples/' + id))); };
$('improvements-body').onclick = event => {
  const run = event.target.closest('[data-open-run]')?.dataset.openRun;
  const candidate = event.target.closest('[data-open-candidate]')?.dataset.openCandidate;
  const job = event.target.closest('[data-job]');
  if (run) action(async () => { await showRun(run); switchTab('results'); });
  if (candidate) action(() => showCandidate(candidate));
  if (job) action(async () => { const detail = await api('improvements/' + job.dataset.job); job.nextElementSibling.textContent = JSON.stringify(detail,null,2); });
};
$('sample-form').onsubmit = event => { event.preventDefault(); action(async () => {
  const data = readForm(event.target, ['manual_probability','manual_low','manual_high']);
  if (state.sample) data.revision = state.sample.revision;
  await api('samples' + (state.sample ? '/' + state.sample.id : ''), data);
  $('sample-dialog').close(); await loadLists();
}, $('save-sample')); };
$('run-form').onsubmit = event => { event.preventDefault(); action(async () => {
  const data = readForm(event.target,['max_output_tokens','percent','seed','threshold']);
  data.reference_run_id ||= null;
  data.sample_ids = [...$('manual-selection').querySelectorAll('input:checked')].map(el => el.value);
  const run = await api('runs',data); await loadLists(); await showRun(run.id); switchTab('results');
}, $('start-run')); };
$('preview-prompt').onclick = () => action(async () => { const p = await api('prompt?id=' + encodeURIComponent($('run-prompt').value)); $('text-heading').textContent = p.name; $('text-content').textContent = p.text; $('text-dialog').showModal(); });
$('cancel-run').onclick = () => action(async () => { if (state.run) { await api('runs/' + state.run.id + '/cancel',{}); await showRun(state.run.id); } });
$('export-csv').onclick = () => action(async () => { if (state.run) await download('runs/' + state.run.id + '/csv',state.run.id + '.csv'); });
$('improve-form').onsubmit = event => { event.preventDefault(); action(async () => {
  if (!state.run) return;
  await api('improvements',{...readForm(event.target,['max_output_tokens']), run_id:state.run.id});
  await loadLists(); switchTab('candidates');
}, event.submitter); };
$('candidate-form').onsubmit = event => { event.preventDefault(); action(async () => {
  const c = await api('candidates',{...readForm(event.target), parent_id:state.candidate});
  await loadLists(); await showCandidate(c.id);
}, event.submitter); };
$('rename-form').onsubmit = event => { event.preventDefault(); action(async () => {
  await api('candidates/' + state.candidate + '/rename', readForm(event.target));
  await loadLists();
}, event.submitter); };
$('remove-version').onclick = () => action(async () => {
  const name = $('version-name').value.trim();
  if (!confirm(`Remove "${name}"? Runs that already used it keep their saved copy of the prompt.`)) return;
  await api('candidates/' + state.candidate + '/delete', {});
  await loadLists(); await showCandidate('repo');
}, $('remove-version'));
$('use-candidate').onclick = () => { $('run-prompt').value = state.candidate; switchTab('evaluate'); };
$('download-prompt').onclick = () => action(async () => {
  if (state.candidate === 'repo') { const p = await api('prompt'); const url = URL.createObjectURL(new Blob([p.text])); const a = document.createElement('a'); a.href=url; a.download='ai_origin.txt'; a.click(); URL.revokeObjectURL(url); }
  else await download('candidates/' + state.candidate + '/txt',state.candidate + '.txt');
});
$('access-form').onsubmit = event => { event.preventDefault(); sessionStorage.setItem('ai-origin-token',$('access-token').value); $('access-token').value=''; action(init); };

async function init() {
  const config = await api('config');
  $('storage').textContent = config.data_dir;
  $('tracing').textContent = `LangGraph · LangSmith ${config.tracing_enabled ? 'on · ' + config.langsmith_project : 'off'}`;
  $('connection').textContent = config.openai_configured ? 'Loading OpenAI models…' : 'Set CHAT_GPT_KEY on the server to evaluate';
  await loadLists(); await showCandidate('repo');
  const query = new URLSearchParams(location.search), runId = query.get('run'), sampleId = query.get('sample');
  if (runId) { await showRun(runId); switchTab('results'); }
  if (sampleId) {
    if (runId) { const row = state.run.results.find(r => r.sample.id === sampleId); if (!row) throw new Error('Sample is not part of this run.'); editSample(row.sample,true); }
    else editSample(await api('samples/' + sampleId));
  }
  if (config.openai_configured) await loadModels();
}

let polling = false;
setInterval(async () => {
  if (polling || document.hidden || ![...state.runs,...state.improvements].some(j => active(j.status))) return;
  polling = true;
  try { await loadLists(); if (state.run) await showRun(state.run.id); }
  catch (error) { notice(error.message,true); }
  finally { polling = false; }
},3000);
action(init);
