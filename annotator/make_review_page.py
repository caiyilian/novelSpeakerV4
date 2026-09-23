# -*- coding: utf-8 -*-
"""Generate an interactive HTML review page for the vol1 error list.

Each item is a 'true/false' judgment card:
  - shows the quote, the answer label, the model label, and raw context
  - the reviewer picks: 答案对 / 答案错 / 存疑
  - optionally types the corrected label
  - can export all decisions as JSON (localStorage-persisted)
No external CDN; single self-contained file.
"""
import json, pathlib

ROOT = pathlib.Path('E:/projects/novelSpeakerV4')
v = 1
data = json.loads((ROOT / 'docs' / f'审阅_vol{v}_文件A_待核清单.json').read_text(encoding='utf-8'))
items = data['items']
canon = data['canon']

payload = json.dumps({'volume': v, 'canon': canon, 'items': items}, ensure_ascii=False)

html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>第 __V__ 卷 标注审阅</title>
<style>
:root{
  --bg:#0f1115; --panel:#171a21; --panel2:#1e222b; --line:#2a2f3a;
  --fg:#e6e8ee; --dim:#9aa3b2; --accent:#6ea8fe; --ok:#3ddc97; --bad:#ff6b6b; --warn:#ffc857;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.7 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
header{position:sticky;top:0;z-index:10;background:rgba(15,17,21,.95);backdrop-filter:blur(8px);
  border-bottom:1px solid var(--line);padding:12px 20px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
h1{font-size:16px;margin:0;font-weight:600}
.stat{color:var(--dim);font-size:13px}
.stat b{color:var(--fg)}
.spacer{flex:1}
button{background:var(--panel2);color:var(--fg);border:1px solid var(--line);border-radius:8px;
  padding:7px 14px;cursor:pointer;font-size:13px;transition:.15s}
button:hover{border-color:var(--accent);color:var(--accent)}
button.primary{background:var(--accent);color:#0b0d12;border-color:var(--accent);font-weight:600}
button.primary:hover{opacity:.88;color:#0b0d12}
main{max-width:1080px;margin:0 auto;padding:20px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px;margin-bottom:16px}
.card.done{border-color:#2f6b4f}
.card-head{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap;margin-bottom:12px}
.id{font-family:ui-monospace,Consolas,monospace;color:var(--accent);font-weight:600}
.kind{font-size:11px;padding:2px 8px;border-radius:999px;border:1px solid var(--line);color:var(--dim)}
.kind.A2{border-color:#6b5a2f;color:var(--warn)}
.quote{background:var(--panel2);border-left:3px solid var(--accent);padding:10px 14px;border-radius:0 8px 8px 0;
  margin:10px 0;font-size:15px}
.labels{display:flex;gap:20px;flex-wrap:wrap;margin:12px 0}
.label-item{font-size:13px}
.label-item .k{color:var(--dim)}
.tag{display:inline-block;padding:2px 9px;border-radius:6px;background:var(--panel2);border:1px solid var(--line);
  font-family:ui-monospace,Consolas,monospace}
.tag.ans{border-color:#3a4a6b}
.tag.model{border-color:#6b4a3a}
details{margin:12px 0}
summary{cursor:pointer;color:var(--dim);font-size:13px;user-select:none}
summary:hover{color:var(--accent)}
pre{background:#0b0d12;border:1px solid var(--line);border-radius:8px;padding:12px;overflow-x:auto;
  font:12px/1.6 ui-monospace,Consolas,monospace;color:#c8cfdb;white-space:pre-wrap;word-break:break-word}
pre .hl{color:var(--warn);font-weight:600}
.choices{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-top:14px;padding-top:14px;border-top:1px solid var(--line)}
.choice{padding:7px 16px;border-radius:8px;border:1px solid var(--line);background:var(--panel2);cursor:pointer;font-size:13px}
.choice:hover{border-color:var(--accent)}
.choice.on-ok{background:rgba(61,220,151,.15);border-color:var(--ok);color:var(--ok);font-weight:600}
.choice.on-bad{background:rgba(255,107,107,.15);border-color:var(--bad);color:var(--bad);font-weight:600}
.choice.on-warn{background:rgba(255,200,87,.15);border-color:var(--warn);color:var(--warn);font-weight:600}
input[type=text]{background:var(--panel2);border:1px solid var(--line);border-radius:8px;color:var(--fg);
  padding:7px 12px;font-size:13px;min-width:180px}
input[type=text]:focus{outline:none;border-color:var(--accent)}
.note{color:var(--dim);font-size:12px;margin-top:10px}
#done{position:fixed;right:24px;bottom:24px;z-index:20;display:none}
#done button{box-shadow:0 8px 24px rgba(0,0,0,.4)}
</style>
</head>
<body>
<header>
  <h1>第 __V__ 卷 · 标注审阅</h1>
  <span class="stat">共 <b id="total">0</b> 条 · 已判 <b id="nDone">0</b></span>
  <span class="stat">答案对 <b id="nOk" style="color:var(--ok)">0</b> · 答案错 <b id="nBad" style="color:var(--bad)">0</b> · 存疑 <b id="nWarn" style="color:var(--warn)">0</b></span>
  <span class="spacer"></span>
  <button onclick="filterCards('all')">全部</button>
  <button onclick="filterCards('undone')">未判</button>
  <button onclick="exportJSON()" class="primary">导出判断</button>
</header>
<main id="list"></main>
<div id="done"><button class="primary" onclick="exportJSON()">已全部判完，导出 →</button></div>
<script>
const DATA = __PAYLOAD__;
const KEY = 'review_vol' + DATA.volume;
let state = JSON.parse(localStorage.getItem(KEY) || '{}');

function esc(s){return (s||'').replace(/[&<>]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function hlCtx(ctx, line){
  return esc(ctx).split('\\n').map(l=>{
    if (l.includes('>>>')) return '<span class="hl">'+l+'</span>';
    return l;
  }).join('\\n');
}

function render(){
  const list = document.getElementById('list');
  list.innerHTML = '';
  DATA.items.forEach((it, i)=>{
    const st = state[it.d] || {};
    const card = document.createElement('div');
    card.className = 'card' + (st.verdict ? ' done' : '');
    card.dataset.d = it.d;
    card.dataset.done = st.verdict ? '1' : '0';
    card.innerHTML = `
      <div class="card-head">
        <span class="id">D${String(it.d).padStart(4,'0')}</span>
        <span class="kind ${it.kind}">${it.kind === 'A1' ? '模型≠答案' : '答案多标签'}</span>
        <span class="kind">第 ${it.line} 行</span>
      </div>
      <div class="quote">${esc(it.quote)}</div>
      <div class="labels">
        <span class="label-item"><span class="k">参考答案：</span><span class="tag ans">${esc(it.answer.join(' / '))}</span></span>
        <span class="label-item"><span class="k">模型输出：</span><span class="tag model">${esc(it.model || '(空)')}</span></span>
      </div>
      <details><summary>展开原文上下文</summary><pre>${hlCtx(it.context, it.line)}</pre></details>
      <div class="choices">
        <span style="color:var(--dim);font-size:13px">判断：</span>
        <div class="choice ${st.verdict==='ok'?'on-ok':''}" onclick="setV(${it.d},'ok',${i})">答案对</div>
        <div class="choice ${st.verdict==='bad'?'on-bad':''}" onclick="setV(${it.d},'bad',${i})">答案错</div>
        <div class="choice ${st.verdict==='warn'?'on-warn':''}" onclick="setV(${it.d},'warn',${i})">存疑</div>
        <input type="text" placeholder="改判为（可选）" value="${esc(st.fix||'')}"
               oninput="setFix(${it.d}, this.value, ${i})">
      </div>
    `;
    list.appendChild(card);
  });
  updateStats();
}

function setV(d, v, i){
  state[d] = state[d] || {};
  state[d].verdict = (state[d].verdict === v) ? null : v;
  save(); render();
}
function setFix(d, val, i){
  state[d] = state[d] || {};
  state[d].fix = val;
  save();
}
function save(){ localStorage.setItem(KEY, JSON.stringify(state)); }
function updateStats(){
  const all = DATA.items.length;
  let ok=0,bad=0,warn=0,done=0;
  DATA.items.forEach(it=>{
    const st = state[it.d];
    if(st && st.verdict){ done++; if(st.verdict==='ok')ok++; else if(st.verdict==='bad')bad++; else warn++; }
  });
  document.getElementById('total').textContent = all;
  document.getElementById('nDone').textContent = done;
  document.getElementById('nOk').textContent = ok;
  document.getElementById('nBad').textContent = bad;
  document.getElementById('nWarn').textContent = warn;
  document.getElementById('done').style.display = (done===all && all>0) ? 'block' : 'none';
}
function filterCards(mode){
  document.querySelectorAll('.card').forEach(c=>{
    c.style.display = (mode==='undone' && c.dataset.done==='1') ? 'none' : '';
  });
}
function exportJSON(){
  const out = { volume: DATA.volume, exported: new Date().toISOString(),
    decisions: DATA.items.map(it=>({
      d: it.d, kind: it.kind, line: it.line, quote: it.quote,
      answer: it.answer, model: it.model,
      verdict: (state[it.d]||{}).verdict || null,
      fix: (state[it.d]||{}).fix || ''
    })) };
  const blob = new Blob([JSON.stringify(out, null, 2)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'vol' + DATA.volume + '_人工审阅结果.json';
  a.click();
}
render();
</script>
</body>
</html>
"""

html = html.replace('__V__', str(v)).replace('__PAYLOAD__', payload)
p = ROOT / 'docs' / f'审阅_vol{v}_人工审阅版.html'
p.write_text(html, encoding='utf-8')
print(f'已写出 {p}')
print(f'  {len(items)} 条判断题，{len(html)} 字符')
