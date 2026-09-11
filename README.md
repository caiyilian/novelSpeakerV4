# NovelSpeakerV4

Multi-agent novel dialogue speaker annotation system using Ollama + Qwen3:32B.

## What It Does

Given a novel text, automatically annotates each `「dialogue」` with the correct speaker name:

```
Input  (novel.txt)
「这是最后一件了吧？」
「嗯，这里确实有……七十件。多谢惠顾。」

Output  (labeled.txt)
村民
罗伦斯
```

## Architecture

```
Python (orchestrator) → Boss → Labeler (with tool calling)
                              → ShortMem (recent N rounds)
                              → LongMem (progressive compression)
Python ← result → write_label.py (1 at a time)
```

**Key design principles:**
1. Python controls the flow — Agents are read-only, never write
2. One dialogue per round — quality over speed
3. Each Agent has **fully independent context** (separate API calls)
4. Short-term memory: last N rounds of annotation summaries
5. Long-term memory: progressive compression of short-term history
6. Labeler has `read_novel_lines` tool — can search the novel on demand

## Why Multi-Agent?

Single agent annotating 1,349 dialogues with no memory between rounds loses context:
- Can't track identity changes ("girl" → "赫萝")
- Can't maintain character relationships
- Re-infers everything from scratch each round

Multi-agent with memory solves this:
- **Boss context: ~2,930 tokens (27%)** — only receives summaries
- **Worker context: ~2,000 tokens each** — independent, parallel-safe
- **ShortMem**: tracks recent N rounds
- **LongMem**: compresses history like exponential moving average — recent info stays detailed, old info gets compressed

## Project Structure

```
novelSpeakerV4/
├── novel.txt                  # Novel text (source)
├── answers.txt                # Ground truth for validation
├── labeled.txt                # Annotation output (generated)
├── label_log.jsonl            # Detailed per-round logs
├── ip_config                  # Ollama server config
│
├── run_label.py               # Main annotation script (multi-agent)
├── get_dialogue.py            # Dialogue extraction
├── write_label.py             # Write one annotation
├── read_log.py                # Parse and display logs
│
├── OLLAMA_TOKEN_STATS.md     # API token field reference
└── 方案.md                    # Architecture design document
```

## Requirements

- Python 3.6+
- `requests` library
- Ollama server with Qwen3:32B model

## Usage

### Configure

Edit `ip_config`:
```
OLLAMA_BASE_URL=http://your-server:11434
OLLAMA_MODEL=qwen3:32b
```

### Run Annotation

```bash
# Annotate 1 dialogue
python run_label.py --count 1

# Annotate 10 dialogues with validation
python run_label.py --count 10 --validate

# Start from dialogue #5
python run_label.py --start 5 --count 10

# Parameters
python run_label.py \
  --count 10 \
  --context-range 20 \
  --short-mem 5 \
  --long-mem-every 5 \
  --validate
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--start` | 0 (resume) | Start from dialogue index |
| `--count` | 1 | Number of dialogues to annotate |
| `--context-range` | 20 | Context window ±lines |
| `--short-mem` | 5 | Short-term memory rounds |
| `--long-mem-every` | 5 | Long-term memory compression frequency |
| `--validate` | false | Run validation after annotation |

### View Logs

```bash
python read_log.py
```

Log format: one JSON per line in `label_log.jsonl`, each containing:
- Round number, dialogue line/text
- Per-agent: input messages, response, token counts
- Tool calls: function name, arguments, results
- Final annotation result

### Validation

```bash
python run_label.py --count 10 --validate
```

Compares `labeled.txt` against `answers.txt` and reports accuracy.

## Memory Mechanism

### Short-Term Memory

Maintains last N rounds of annotation summaries:
```
[短期记忆 - 最近标注]
  #37: 罗伦斯 | 看到修道院有人挥手
  #41: 罗伦斯 | 发现是骑士
  #43: 骑士 → 罗伦斯 | 盘问身份
```

### Long-Term Memory

Progressive compression of short-term history:
- **Keeps**: character relationships, main plot, key settings, alias mappings
- **Drops**: specific dialogue content, transaction details, minor characters
- **Strategy**: like exponential moving average — recent rounds keep more detail, older rounds get compressed harder

## Test Results (First 10 Dialogues)

| Metric | Value |
|--------|-------|
| Total dialogues | 1,349 |
| First 10 accuracy | 90.0% (9/10) |
| Total token consumption | 21,443 |
| Tool calls | 0 |
| Long-term memory compressions | 2 |

Note: First 4 dialogues are particularly difficult (narration mixed with dialogue, no explicit speaker names). From dialogue #5 onward: 100% accuracy.

## License

MIT

## Desktop Control Center

Install the desktop dependency once, then launch the five-volume task center:

```cmd
python -m pip install -r requirements-gui.txt
launch_control_center.cmd
```

The first launch requires selecting the SenseNova API key file. The control center
stores only the selected path, starts each volume as a hidden child process, and
keeps running in the Windows system tray when the window is closed. Continue and
retry preserve the current checkpoint. Restart creates a deduplicated backup before
passing `--reset-state` to the annotation process.

While a volume is running, its row shows the active first-pass or full-volume
review stage and an estimated remaining time. First-pass ETA comes from the
annotator's measured throughput with a live fallback estimate; review ETA is
restored from its durable checkpoint after retries or interruptions.

### Windows EXE

Build the one-file Windows control center with:

```cmd
build_control_center_exe.cmd
```

The output is `dist\NovelSpeakerControlCenter.exe`. It contains both the GUI and
the annotation worker. Keep the EXE in the repository root (or in its `dist`
folder) so it can find the existing `data` and `config` directories. API keys and
runtime data are never bundled into the executable.
