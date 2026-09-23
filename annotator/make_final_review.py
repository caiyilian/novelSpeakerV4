# -*- coding: utf-8 -*-
"""Generate the FINAL human-review page: merges my judgment + the large model's,
highlights the disagreements, and lets the user decide each item."""
import json, pathlib, re, sys, importlib.util

ROOT = pathlib.Path('E:/projects/novelSpeakerV4')
spec = importlib.util.spec_from_file_location('sf', ROOT / 'annotator' / 'score_fixed.py')
sf = importlib.util.module_from_spec(spec); spec.loader.exec_module(sf)

v = 1
novel = (sf.VOL[v] / 'novel.txt').read_text(encoding='utf-8').splitlines()
A = json.loads((ROOT / 'docs' / f'审阅_vol{v}_文件A_待核清单.json').read_text(encoding='utf-8'))
items = A['items']
canon = A['canon']

# 大模型结果
lm = {}
for line in (ROOT / 'docs' / f'审阅_vol{v}_大模型审核结果_2026-09-24.txt').read_text(encoding='utf-8').splitlines():
    line = line.strip()
    if not line: continue
    p = [x.strip() for x in line.split('|')]
    if len(p) >= 3:
        lm[p[0]] = {'verdict': p[1], 'label': p[2], 'evidence': p[3] if len(p) > 3 else ''}

# 我的判断（含最终裁决：经原文核实后我采纳谁的）
# kind: agree=双方一致; lm_fix=大模型对(我已核实); me_fix=我对(大模型错); me_unc=我存疑; lm_unc=大模型存疑
MINE = {
 'D0001': ('存疑', '?', 'me_unc'), 'D0002': ('存疑', '?', 'me_unc'),
 'D0003': ('模型错', '村民', 'agree'),
 'D0045': ('模型错', '无人称引语', 'agree'),
 'D0046': ('存疑', '?', 'me_unc'),
 'D0174': ('模型错', '无人称引语', 'agree'),
 'D0191': ('模型错', '赫萝', 'agree'),
 'D0239': ('模型错', '罗伦斯', 'agree'),
 'D0290': ('存疑', '?', 'me_unc'), 'D0291': ('存疑', '?', 'me_unc'),
 'D0328': ('模型错', '赫萝', 'agree'),
 'D0476': ('模型错', '无人称引语', 'agree'),
 'D0540': ('模型错', '杰廉', 'agree'), 'D0541': ('模型错', '罗伦斯', 'agree'),
 'D0581': ('模型错', '工匠', 'agree'),
 'D0596': ('模型错', '赫萝', 'agree'), 'D0597': ('模型错', '罗伦斯', 'agree'),
 'D0629': ('存疑', '?', 'me_unc'),
 'D0665': ('模型错', '罗伦斯', 'agree'), 'D0666': ('模型错', '怀兹', 'agree'),
 'D0728': ('模型错', '无人称引语', 'agree'),
 'D0782': ('模型错', '罗伦斯', 'agree'),
 'D0796': ('模型错', '店老板', 'agree'),
 'D0911': ('模型错', '无人称引语', 'me_fix'),      # 我对，大模型错
 'D0914': ('模型错', '赫萝', 'agree'),
 'D0991': ('模型错', '无人称引语', 'agree'),
 'D0998': ('模型错', '商行手下', 'lm_fix'),        # 大模型对（账本 canonical）
 'D1000': ('模型错', '商行手下', 'lm_fix'),
 'D1023': ('答案错', '梅迪欧商行手下', 'lm_fix'),  # 大模型对（L2174 回指）
 'D1081': ('模型错', '无人称引语', 'me_fix'),      # 我对（暗号）
 'D1082': ('模型错', '无人称引语', 'me_fix'),
 'D1083': ('模型错', '商行手下', 'lm_fix'),
 'D1084': ('模型错', '商行手下', 'lm_fix'),
 'D1086': ('模型错', '商行手下', 'lm_fix'),
 'D1087': ('模型错', '商行手下', 'lm_fix'),
 'D1101': ('模型错', '赫萝', 'me_fix'),            # 我对（L2362 赫萝这么说）
 'D1126': ('模型错', '无人称引语', 'agree'),
 'D1143': ('模型错', '罗伦斯', 'agree'),
 'D1167': ('模型错', '赫萝', 'agree'), 'D1168': ('模型错', '赫萝', 'agree'),
 'D1169': ('模型错', '罗伦斯', 'agree'),
 'D1183': ('模型错', '赫萝', 'agree'), 'D1184': ('模型错', '罗伦斯', 'agree'),
 'D1307': ('模型错', '马贺特', 'agree'), 'D1319': ('模型错', '马贺特', 'agree'),
 'D1342': ('答案错', '商行手下', 'agree'),
}

CTX = 6
def ctx(ln):
    lo, hi = max(1, ln - CTX), min(len(novel), ln + CTX)
    return "\n".join(f"{k:>5}{' >>>' if k == ln else '    '} {novel[k-1]}" for k in range(lo, hi + 1))

merged = []
seen = set()
for it in items:
    if it['d'] in seen:
        continue          # A1/A2 段可能重复收录同一 D 编号，去重
    seen.add(it['d'])
    d = 'D%04d' % it['d']
    mine = MINE.get(d)
    l = lm.get(d, {})
    kind = mine[2] if mine else 'agree'
    merged.append({
        'd': it['d'], 'line': it['line'], 'quote': it['quote'],
        'answer': it['answer'], 'model': it['model'],
        'context': ctx(it['line']),
        'mine_v': mine[0] if mine else '', 'mine_label': mine[1] if mine else '',
        'lm_v': l.get('verdict', ''), 'lm_label': l.get('label', ''), 'lm_ev': l.get('evidence', ''),
        'kind': kind,
    })

# 分歧项排前
order = {'lm_fix': 0, 'me_fix': 1, 'me_unc': 2, 'agree': 3}
merged.sort(key=lambda x: (order.get(x['kind'], 9), x['d']))

payload = json.dumps({'volume': v, 'canon': canon, 'items': merged}, ensure_ascii=False)

html = r"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>第 __V__ 卷 · 最终人工审阅</title>
<style>
:root{--bg:#0f1115;--panel:#171a21;--panel2:#1e222b;--line:#2a2f3a;--fg:#e6e8ee;--dim:#9aa3b2;
--accent:#6ea8fe;--ok:#3ddc97;--bad:#ff6b6b;--warn:#ffc857;--purple:#c792ea}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.7 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
header{position:sticky;top:0;z-index:10;background:rgba(15,17,21,.96);backdrop-filter:blur(8px);
border-bottom:1px solid var(--line);padding:12px 20px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
h1{font-size:16px;margin:0;font-weight:600}
.stat{color:var(--dim);font-size:13px}.stat b{color:var(--fg)}
.spacer{flex:1}
button{background:var(--panel2);color:var(--fg);border:1px solid var(--line);border-radius:8px;
padding:7px 13px;cursor:pointer;font-size:13px;transition:.15s}
button:hover{border-color:var(--accent);color:var(--accent)}
button.primary{background:var(--accent);color:#0b0d12;border-color:var(--accent);font-weight:600}
main{max-width:1120px;margin:0 auto;padding:20px}
.banner{background:linear-gradient(90deg,rgba(199,146,234,.12),transparent);border:1px solid #3a2f4a;
border-radius:10px;padding:12px 16px;margin-bottom:18px;font-size:13px;color:var(--dim)}
.banner b{color:var(--purple)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px;margin-bottom:14px}
.card.disagree{border-left:4px solid var(--purple)}
.card.done{border-color:#2f6b4f}
.card-head{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap;margin-bottom:10px}
.id{font-family:ui-monospace,Consolas,monospace;color:var(--accent);font-weight:600}
.chip{font-size:11px;padding:2px 9px;border-radius:999px;border:1px solid var(--line);color:var(--dim)}
.chip.disagree{background:rgba(199,146,234,.15);border-color:var(--purple);color:var(--purple)}
.quote{background:var(--panel2);border-left:3px solid var(--accent);padding:10px 14px;border-radius:0 8px 8px 0;margin:10px 0;font-size:15px}
.cmp{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:12px 0}
@media(max-width:720px){.cmp{grid-template-columns:1fr}}
.side{background:var(--panel2);border:1px solid var(--line);border-radius:9px;padding:11px 14px}
.side .who{font-size:12px;color:var(--dim);margin-bottom:6px}
.side .verdict{font-weight:600;margin-bottom:4px}
.side .lab{font-family:ui-monospace,Consolas,monospace;color:var(--warn)}
.side .ev{font-size:12px;color:var(--dim);margin-top:6px;line-height:1.5}
.side.mine{border-color:#3a4a6b}.side.lm{border-color:#6b4a3a}
.raw{font-size:13px;color:var(--dim)}
details{margin:10px 0}
summary{cursor:pointer;color:var(--dim);font-size:13px}
summary:hover{color:var(--accent)}
pre{background:#0b0d12;border:1px solid var(--line);border-radius:8px;padding:12px;overflow-x:auto;
font:12px/1.6 ui-monospace,Consolas,monospace;color:#c8cfdb;white-space:pre-wrap;word-break:break-word}
pre .hl{color:var(--warn);font-weight:600}
.choices{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-top:12px;padding-top:12px;border-top:1px solid var(--line)}
.choice{padding:7px 15px;border-radius:8px;border:1px solid var(--line);background:var(--panel2);cursor:pointer;font-size:13px}
.choice:hover{border-color:var(--accent)}
.choice.on-ok{background:rgba(61,220,151,.15);border-color:var(--ok);color:var(--ok);font-weight:600}
.choice.on-bad{background:rgba(255,107,107,.15);border-color:var(--bad);color:var(--bad);font-weight:600}
.choice.on-warn{background:rgba(255,200,87,.15);border-color:var(--warn);color:var(--warn);font-weight:600}
input[type=text]{background:var(--panel2);border:1px solid var(--line);border-radius:8px;color:var(--fg);
padding:7px 12px;font-size:13px;min-width:170px}
input[type=text]:focus{outline:none;border-color:var(--accent)}
</style></head><body>
<header>
  <h1>第 __V__ 卷 · 最终人工审阅</h1>
  <span class="stat">共 <b id="total">0</b> · 已决 <b id="nDone">0</b></span>
  <span class="stat">分歧 <b id="nDis" style="color:var(--purple)">0</b></span>
  <span class="spacer"></span>
  <button onclick="filt('all')">全部</button>
  <button onclick="filt('disagree')">只看分歧</button>
  <button onclick="filt('undone')">未决</button>
  <button class="primary" onclick="exp()">导出决定</button>
</header>
<main>
  <div class="banner">
    <b>审阅说明</b>：每条列出 <b>小灵</b>（AI 助手）与 <b>大模型</b> 两个独立判断。左侧紫边为<b>双方有分歧</b>的条目，已按「经原文核实后谁对」标注。请对你的最终决定勾选。
  </div>
  <div id="list"></div>
</main>
<script>
const DATA = __PAYLOAD__;
const KEY = 'final_review_vol' + DATA.volume;
let state = JSON.parse(localStorage.getItem(KEY) || '{}');
const esc = s => (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const KIND = {lm_fix:['大模型对（我漏查）','disagree'],me_fix:['我对（大模型错）','disagree'],
  me_unc:['我存疑','disagree'],agree:['双方一致','']};

function render(){
  const list = document.getElementById('list'); list.innerHTML='';
  DATA.items.forEach(it=>{
    const st = state[it.d] || {};
    const [ktext, kcls] = KIND[it.kind] || ['',''];
    const div = document.createElement('div');
    div.className = 'card' + (kcls?' disagree':'') + (st.verdict?' done':'');
    div.dataset.done = st.verdict ? '1':'0';
    div.dataset.dis = kcls ? '1':'0';
    div.innerHTML = `
      <div class="card-head">
        <span class="id">D${String(it.d).padStart(4,'0')}</span>
        <span class="chip">第 ${it.line} 行</span>
        ${ktext?`<span class="chip ${kcls}">${ktext}</span>`:''}
      </div>
      <div class="quote">${esc(it.quote)}</div>
      <div class="raw">参考答案：<span class="lab">${esc(it.answer.join(' / '))}</span> ｜ 模型输出：<span class="lab">${esc(it.model||'(空)')}</span></div>
      <div class="cmp">
        <div class="side mine"><div class="who">小灵（AI 助手）</div>
          <div class="verdict">${esc(it.mine_v||'—')}</div>
          <div>标签：<span class="lab">${esc(it.mine_label||'—')}</span></div></div>
        <div class="side lm"><div class="who">大模型</div>
          <div class="verdict">${esc(it.lm_v||'—')}</div>
          <div>标签：<span class="lab">${esc(it.lm_label||'—')}</span></div>
          <div class="ev">依据：${esc(it.lm_ev||'—')}</div></div>
      </div>
      <details><summary>展开原文上下文</summary><pre>${esc(it.context).split('\n').map(l=>l.includes('>>>')?`<span class="hl">${l}</span>`:l).join('\n')}</pre></details>
      <div class="choices">
        <span style="color:var(--dim);font-size:13px">你的决定：</span>
        <div class="choice ${st.verdict==='ok'?'on-ok':''}" onclick="sv(${it.d},'ok')">答案对</div>
        <div class="choice ${st.verdict==='bad'?'on-bad':''}" onclick="sv(${it.d},'bad')">答案错</div>
        <div class="choice ${st.verdict==='warn'?'on-warn':''}" onclick="sv(${it.d},'warn')">存疑</div>
        <input type="text" placeholder="最终标签（可选）" value="${esc(st.fix||'')}" oninput="sf2(${it.d},this.value)">
      </div>`;
    list.appendChild(div);
  });
  stats();
}
function sv(d,v){ state[d]=state[d]||{}; state[d].verdict = state[d].verdict===v?null:v; save(); render(); }
function sf2(d,val){ state[d]=state[d]||{}; state[d].fix=val; save(); }
function save(){ localStorage.setItem(KEY, JSON.stringify(state)); }
function stats(){
  let done=0,dis=0;
  DATA.items.forEach(it=>{ if(state[it.d]&&state[it.d].verdict) done++; if(KIND[it.kind][1]) dis++; });
  document.getElementById('total').textContent=DATA.items.length;
  document.getElementById('nDone').textContent=done;
  document.getElementById('nDis').textContent=dis;
}
function filt(m){ document.querySelectorAll('.card').forEach(c=>{
  if(m==='disagree') c.style.display = c.dataset.dis==='1'?'':'none';
  else if(m==='undone') c.style.display = c.dataset.done==='1'?'none':'';
  else c.style.display='';
});}
function exp(){
  const out={volume:DATA.volume,exported:new Date().toISOString(),
    decisions:DATA.items.map(it=>({d:it.d,line:it.line,quote:it.quote,answer:it.answer,model:it.model,
      mine:it.mine_v+'/'+it.mine_label, lm:it.lm_v+'/'+it.lm_label, kind:it.kind,
      verdict:(state[it.d]||{}).verdict||null, final_label:(state[it.d]||{}).fix||''}))};
  const b=new Blob([JSON.stringify(out,null,2)],{type:'application/json'});
  const a=document.createElement('a'); a.href=URL.createObjectURL(b);
  a.download='vol'+DATA.volume+'_最终审阅决定.json'; a.click();
}
render();
</script></body></html>
"""

html = html.replace('__V__', str(v)).replace('__PAYLOAD__', payload)
p = ROOT / 'docs' / f'审阅_vol{v}_最终人工审阅版.html'
p.write_text(html, encoding='utf-8')
print(f'已写出 {p}')
print(f'  {len(merged)} 条，其中分歧 {sum(1 for x in merged if x["kind"]!="agree")} 条')
