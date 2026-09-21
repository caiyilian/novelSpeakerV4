import sys,pathlib,re,json,collections,hashlib
sys.stdout.reconfigure(encoding='utf-8')
root=pathlib.Path.cwd();sys.path.insert(0,str(root/'src'));import run_label as rl
base=root/'tmp/deepseek_probe/results';backup=root/'backup/2026-09-01_205124_post_native256k_full_volume_review_vol1-5';out=root/'tmp/agent_survey/20260921'
def parse(p):
 d={}
 for l in p.read_text(encoding='utf-8').splitlines():
  m=re.match(r'^D?(\d+)\s*[|｜:：\t]\s*(.*)$',l.strip())
  if m:d[int(m[1])]=m[2].strip()
 return d
def parts(x):return {'非人物发声' if y.strip()=='无人称引语' else y.strip() for y in x.split('|') if y.strip()}
rows=[];allc=collections.Counter();hashes={}
for v in range(1,6):
 vd=root/'data' if v==1 else root/f'data/volume{v}';ans=[parts(x) for x in re.findall(r'【([^】]+)】',(vd/'answers.txt').read_text(encoding='utf-8'))]
 ids=rl._load_verified_validation_identities(str(backup/f'volume{v}/evidence_vault.json'))
 ct=parse(base/f'baseline_vol{v}_contfull/raw_output.txt');ev=parse(base/f'baseline_vol{v}_evledger/raw_output.txt');old={i:x.strip() for i,x in enumerate((backup/f'volume{v}/labeled.txt').read_text(encoding='utf-8').splitlines(),1)}
 c=collections.Counter();bad=[]
 for i,a in enumerate(ans,1):
  ok=[bool(a&parts(d.get(i,''))) or rl._validation_lenient_match(a,parts(d.get(i,'')),ids)[0] for d in (ct,ev,old)]
  c['total']+=1;c['cont']+=ok[0];c['evledger']+=ok[1];c['old']+=ok[2];c[f'pair_{int(ok[0])}{int(ok[1])}']+=1
  div=parts(ct.get(i,''))!=parts(ev.get(i,''));c['divergent']+=div
  if not any(ok[:2]):
   c['both_wrong_old_correct']+=ok[2];c['both_wrong_divergent']+=div;c['both_wrong_agree']+=not div;bad.append(i)
  c['triple_wrong']+=not any(ok)
 novel=(vd/'novel.txt').read_text(encoding='utf-8').splitlines();counts=[len(re.findall(r'「[^」]*」',l)) for l in novel];c['multi_quote_lines']=sum(x>1 for x in counts);c['quotes_on_multi_lines']=sum(x for x in counts if x>1)
 for tag in ['merge','idres','norm']:
  p=base/f'baseline_vol{v}_{tag}/raw_output.txt'
  if p.exists():
   d=parse(p);c[tag]=sum(bool(a&parts(d.get(i,''))) or rl._validation_lenient_match(a,parts(d.get(i,'')),ids)[0] for i,a in enumerate(ans,1))
 rows.append({'volume':v,**dict(c),'both_wrong_ids':bad});allc.update(c)
 for p in [vd/'novel.txt',vd/'answers.txt',base/f'baseline_vol{v}_contfull/raw_output.txt',base/f'baseline_vol{v}_evledger/raw_output.txt',backup/f'volume{v}/labeled.txt',backup/f'volume{v}/evidence_vault.json']:
  hashes[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
result={'aggregate':dict(allc),'volumes':rows,'hashes':hashes,'scorer_sha256':hashlib.sha256((root/'src/run_label.py').read_bytes()).hexdigest()};(out/'local_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'aggregate':dict(allc),'volumes':[{k:z for k,z in r.items() if k!='both_wrong_ids'} for r in rows]},ensure_ascii=False,indent=2))
