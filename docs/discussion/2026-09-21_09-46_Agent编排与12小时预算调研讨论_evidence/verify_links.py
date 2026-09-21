import concurrent.futures,datetime,json,pathlib,time,urllib.request,urllib.error
base=pathlib.Path('tmp/agent_survey/20260921');refs=json.loads((base/'references.json').read_text(encoding='utf-8'));manifest=json.loads((base/'manifest.json').read_text(encoding='utf-8'))
urls=sorted({r['url'] for r in refs}|{'https://github.com/'+r['repo'] for r in manifest})
def check(url):
 attempts=[]
 for route in ['proxy','direct','proxy']:
  proxy={'http':'http://127.0.0.1:7890','https':'http://127.0.0.1:7890'} if route=='proxy' else {}
  opener=urllib.request.build_opener(urllib.request.ProxyHandler(proxy));start=time.monotonic()
  try:
   req=urllib.request.Request(url.split('#')[0],headers={'User-Agent':'Mozilla/5.0 (research-link-check)','Accept':'text/html'})
   with opener.open(req,timeout=35) as resp:
    body=resp.read();status=resp.status;final=resp.url
   suspicious=any(x in body[:150000].lower() for x in [b'<title>page not found',b'<title>sign in to github',b'<title>just a moment'])
   row={'route':route,'status':status,'final_url':final,'bytes':len(body),'seconds':round(time.monotonic()-start,2),'suspicious_page':suspicious};attempts.append(row)
   if status==200 and len(body)>100 and not suspicious:return {'url':url,'ok':True,'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'attempts':attempts}
  except Exception as e:attempts.append({'route':route,'error':str(e)})
 return {'url':url,'ok':False,'attempts':attempts}
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
 rows=list(ex.map(check,urls))
(base/'link_validation.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
print('checked',len(rows),'passed',sum(r['ok'] for r in rows))
for r in rows:
 if not r['ok']:print(json.dumps(r,ensure_ascii=False))
