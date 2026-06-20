import json

with open(r'E:\projects\GRUMon\label_log.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        entry = json.loads(line)
        print(f'=== Round {entry["round"]} (line {entry["dialogue_line"]}) ===')
        print(f'  Timestamp: {entry["timestamp"]}')
        print(f'  Dialogue: {entry["dialogue_text"][:50]}')
        print(f'  Total tokens: input={entry["total_input_tokens"]}, output={entry["total_output_tokens"]}')
        print(f'  Agents: {list(entry["agents"].keys())}')
        for name, agent in entry['agents'].items():
            print(f'    [{name}] input={agent["prompt_eval_count"]}, output={agent["eval_count"]}, total={agent["total_tokens"]}')
            print(f'      input_summary: {agent["input_summary"][:100]}...')
            print(f'      response: {agent["response"][:200]}...')
        if entry.get('tool_calls'):
            print(f'  Tool calls: {len(entry["tool_calls"])}')
            for tc in entry['tool_calls']:
                print(f'    Round {tc["round"]}: {tc["tool_calls"]}')
        if entry.get('result'):
            print(f'  Result: speaker={entry["result"]["speaker"]}, summary={entry["result"]["summary"][:80]}')
        print()
