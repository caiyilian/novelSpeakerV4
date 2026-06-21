import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
LOG_PATH = os.path.join(ROOT_DIR, 'data', 'label_log.jsonl')

with open(LOG_PATH, 'r', encoding='utf-8') as f:
    entries = [json.loads(line) for line in f]

e = entries[0]
print('=== Entry 0 keys ===')
for k, v in e.items():
    if isinstance(v, dict) and k == 'agents':
        print(f'  agents keys: {list(v.keys())}')
        for agent_name, agent_data in v.items():
            print(f'    {agent_name}:')
            for ak, av in agent_data.items():
                if ak in ('input_messages', 'response'):
                    print(f'      {ak}: {str(av)[:80]}...')
                else:
                    print(f'      {ak}: {av}')
    elif isinstance(v, list) and k == 'tool_calls':
        print(f'  tool_calls: {len(v)} entries')
        for tc in v[:2]:
            print(f'    round={tc.get("round")}, func={[t.get("function") for t in tc.get("tool_calls",[])]}')
    else:
        print(f'  {k}: {v}')

print()
print(f'Total entries: {len(entries)}')
tool_call_entries = sum(1 for e in entries if e.get('tool_calls'))
print(f'Entries with tool_calls: {tool_call_entries}')
longmem_entries = sum(1 for e in entries if 'LongMem' in e.get('agents', {}))
print(f'Entries with LongMem: {longmem_entries}')

# Show tool call details
tc_entries = [e for e in entries if e.get('tool_calls')]
if tc_entries:
    print(f'\n=== Sample tool call entry (first with tools) ===')
    e2 = tc_entries[0]
    for tc in e2['tool_calls']:
        print(f'  round={tc.get("round")}, pec={tc.get("prompt_eval_count")}, ec={tc.get("eval_count")}')
        for t in tc.get('tool_calls', []):
            print(f'    function={t.get("function")}, args={t.get("arguments")}')
