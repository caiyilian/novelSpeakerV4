import concurrent.futures,json,pathlib,subprocess,time
base=pathlib.Path('tmp/agent_survey/20260921');base.mkdir(parents=True,exist_ok=True)
repos={'codex':'openai/codex','opencode':'anomalyco/opencode','lazycodex':'code-yeongyu/lazycodex','deepseek-harness':'deepseek-ai/deepseek-harness','pi':'earendil-works/pi','deepagents':'langchain-ai/deepagents','langgraph':'langchain-ai/langgraph','pydantic-ai':'pydantic/pydantic-ai','mastra':'mastra-ai/mastra','letta':'letta-ai/letta','deer-flow':'bytedance/deer-flow','agno':'agno-agi/agno','openhands':'OpenHands/software-agent-sdk','harness-of-harness':'Flesymeb/HarnessOfHarness','dcp':'Opencode-DCP/opencode-dynamic-context-pruning'}
def clone(kv):
 name,repo=kv;p=base/name
 cmd=['git','-c','http.proxy=http://127.0.0.1:7890','clone','--depth','1','https://github.com/'+repo+'.git',str(p)]
 try:
  r=subprocess.run(cmd,capture_output=True,text=True,timeout=160) if not (p/'.git').exists() else None
  if r and r.returncode:return {'name':name,'repo':repo,'error':r.stderr[-1000:]}
  sha=subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
  date=subprocess.check_output(['git','-C',str(p),'show','-s','--format=%cI','HEAD'],text=True).strip()
  return {'name':name,'repo':repo,'sha':sha,'date':date,'path':str(p)}
 except Exception as e:return {'name':name,'repo':repo,'error':str(e)}
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
 rows=list(ex.map(clone,repos.items()))
(base/'manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
for r in rows: print(json.dumps(r,ensure_ascii=False))
