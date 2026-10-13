'use strict';

// Local SVG charts: no CDN, tracking, or chart-library dependency. Functions use
// saved results only; missing predictions/metrics never become zero values.
window.RunCharts = (() => {
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const number = value => finite(value) ? value.toFixed(1) : '—';
  const series = [
    {key:'mae_pp', label:'MAE', color:'#087c78'},
    {key:'rmse_pp', label:'RMSE', color:'#375bcc'},
    {key:'bias_pp', label:'Signed bias', color:'#b54e16'},
  ];
  const text = (x,y,value,anchor='middle') => `<text x="${x}" y="${y}" text-anchor="${anchor}">${esc(value)}</text>`;
  const line = (x1,y1,x2,y2,cls='chart-grid') => `<line class="${cls}" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"/>`;
  function histogramBins(results) {
    const bins = Array.from({length:20}, (_,i) => ({low:-100+i*10, high:-90+i*10, count:0}));
    for (const r of results) {
      if (r.status !== 'completed' || !finite(r.model_probability) || !finite(r.sample?.manual_probability)) continue;
      const difference = r.model_probability - r.sample.manual_probability;
      if (difference < -100 || difference > 100) continue;
      bins[Math.min(19, Math.floor((difference+100)/10))].count++;
    }
    return bins;
  }
  function histogram(target, run) {
    if (!run) { target.innerHTML = '<p class="help">Select a run to see its differences.</p>'; return; }
    const bins = histogramBins(run.results), total = bins.reduce((sum,b) => sum+b.count,0);
    if (!total) { target.innerHTML = '<p class="help">No successful predictions yet.</p>'; return; }
    const w=760, h=270, left=55, right=20, top=25, bottom=58, plotW=w-left-right, plotH=h-top-bottom;
    const max = Math.max(1,...bins.map(b => b.count)), y = n => top+plotH*(1-n/max), step=plotW/20;
    let marks='';
    for (const n of [...new Set([0,Math.ceil(max/2),max])]) marks += line(left,y(n),w-right,y(n))+text(left-10,y(n)+4,n,'end');
    marks += text(left,14,'Count','start');
    const label = (b,i) => `${b.low} ≤ difference ${i===19?'≤':'<'} ${b.high} pp`;
    bins.forEach((b,i) => {
      marks += `<rect class="histogram-bar" tabindex="0" data-count="${b.count}" x="${left+i*step+1}" y="${y(b.count)}" width="${step-2}" height="${plotH*b.count/max}" fill="#375bcc"><title>${esc(label(b,i))}: ${b.count} samples</title></rect>`;
    });
    for (let d=-100;d<=100;d+=20) marks+=text(left+(d+100)/200*plotW,h-bottom+23,d);
    marks+=line(left+plotW/2,top,left+plotW/2,h-bottom,'chart-zero');
    marks+=text(left+plotW/2,h-9,'Difference (model − manual), percentage points');
    target.innerHTML=`<div class="chart-scroll"><svg class="run-chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="Difference histogram, ${total} successful samples">${marks}</svg></div><p class="help">${total} successful samples · 10-point bins · negative = lower than manual; positive = higher.</p><details><summary>Histogram counts</summary><table><thead><tr><th>Difference interval</th><th>Samples</th></tr></thead><tbody>${bins.map((b,i)=>`<tr><td>${esc(label(b,i))}</td><td>${b.count}</td></tr>`).join('')}</tbody></table></details>`;
  }
  function trendRuns(runs, split) {
    return runs.filter(r => ['completed','completed_with_errors'].includes(r.status) && (split==='all' || r.config.split===(split==='mixed'?'all':split)))
      .sort((a,b) => a.created_at.localeCompare(b.created_at) || a.id.localeCompare(b.id));
  }
  function trends(target, runs, split, selected) {
    const data=trendRuns(runs,split);
    if (!data.length) { target.innerHTML='<p class="help">Complete an evaluation to see metric trends.</p>'; return; }
    const values=data.flatMap(r => series.map(s => r.metrics?.overall?.[s.key])).filter(finite);
    if (!values.length) { target.innerHTML='<p class="help">These runs have no usable agreement metrics.</p>'; return; }
    const low=Math.min(0,Math.floor(Math.min(...values)/10)*10), high=Math.max(10,Math.ceil(Math.max(...values)/10)*10);
    const w=Math.max(680,data.length*45+100), h=290, left=60, right=25, top=25, bottom=60;
    const x=i => data.length===1?(left+w-right)/2:left+i*(w-left-right)/(data.length-1);
    const y=v => top+(high-v)/(high-low)*(h-top-bottom);
    let marks=text(left,14,'Percentage points','start');
    for (let i=0;i<=4;i++) { const v=low+(high-low)*i/4; marks+=line(left,y(v),w-right,y(v))+text(left-10,y(v)+4,number(v),'end'); }
    marks+=line(left,y(0),w-right,y(0),'chart-zero');
    series.forEach(s => {
      // Missing metrics break a line, instead of joining across an absent value.
      let path='', connected=false;
      data.forEach((r,i) => { const v=r.metrics?.overall?.[s.key]; if (!finite(v)) {connected=false;return;} path+=`${connected?'L':'M'}${x(i)},${y(v)} `;connected=true; });
      marks+=`<path class="metric-line" data-metric="${s.key}" d="${path}" fill="none" stroke="${s.color}"/>`;
      data.forEach((r,i) => {
        const m=r.metrics?.overall, v=m?.[s.key]; if (!finite(v)) return;
        const tip=`${r.name} · ${r.id}\n${new Date(r.created_at).toLocaleString()} · ${r.config.model} · ${r.config.split}\n${r.status} · ${m.completed}/${m.total} successful\n${s.label}: ${number(v)} pp`;
        marks+=`<circle class="metric-point" data-run="${esc(r.id)}" data-metric="${s.key}" data-value="${v}" tabindex="0" role="button" aria-label="${esc(tip)}" cx="${x(i)}" cy="${y(v)}" r="${r.id===selected?6:4}" fill="${s.color}" stroke="white" stroke-width="2"><title>${esc(tip)}</title></circle>`;
      });
    });
    data.forEach((r,i)=>{marks+=text(x(i),h-bottom+22,`#${i+1}`);});
    marks+=text((left+w-right)/2,h-10,'Runs in chronological order (oldest → newest)');
    target.innerHTML=`<div class="chart-legend">${series.map(s=>`<span><i style="background:${s.color}"></i>${s.label}</span>`).join('')}</div><div class="chart-scroll"><svg class="run-chart" style="min-width:${w}px" viewBox="0 0 ${w} ${h}" role="group" aria-label="MAE, RMSE, and signed bias across evaluation runs">${marks}</svg></div><details><summary>Run metrics and details</summary><div class="chart-scroll"><table><thead><tr><th>Run</th><th>Dataset / model</th><th>Successful</th>${series.map(s=>`<th>${s.label} (pp)</th>`).join('')}</tr></thead><tbody>${data.map((r,i)=>`<tr><td><button class="secondary" data-run="${esc(r.id)}">#${i+1} · ${esc(r.name)}</button><small>${esc(new Date(r.created_at).toLocaleString())} · ${esc(r.status)}</small></td><td>${esc(r.config.split)} · ${esc(r.config.model)}</td><td>${r.metrics?.overall?.completed??0}/${r.metrics?.overall?.total??0}</td>${series.map(s=>`<td>${number(r.metrics?.overall?.[s.key])}</td>`).join('')}</tr>`).join('')}</tbody></table></div></details>`;
  }
  return {histogramBins, trendRuns, histogram, trends};
})();
