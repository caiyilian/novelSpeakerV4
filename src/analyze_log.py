# -*- coding: utf-8 -*-
"""
分析 label_log.jsonl，生成可视化图表

图表1：每一轮每个 agent 的 token 消耗
图表2：每一轮每个 agent 的工具调用次数
"""
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# 配置中文字体
# ============================================================
plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
LOG_PATH = os.path.join(ROOT_DIR, "data", "label_log.jsonl")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")


def load_logs(path):
    """加载日志文件"""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def extract_agent_data(entries):
    """
    从日志中提取每个 agent 每轮的 token 消耗和工具调用次数
    
    返回:
      agent_names: list[str]  所有出现的 agent 名称
      rounds: list[int]       轮次编号
      token_data: dict[agent_name -> list[int]]  每轮 token 消耗
      tool_data: dict[agent_name -> list[int]]   每轮工具调用次数
    """
    all_agents = set()
    token_data = defaultdict(list)
    tool_data = defaultdict(list)
    rounds = []
    
    for e in entries:
        r = e.get("round", 0)
        rounds.append(r)
        
        # 记录该轮出现的所有 agent 名称
        agents = e.get("agents", {})
        for agent_name in agents:
            all_agents.add(agent_name)
        
        # 记录该轮没有出现的 agent 用 0 占位
        # 我们稍后统一填充
    
    # 为每个 agent 初始化列表
    sorted_agents = sorted(all_agents)
    for agent in sorted_agents:
        token_data[agent] = [0] * len(entries)
        tool_data[agent] = [0] * len(entries)
    
    # 填充数据
    for i, e in enumerate(entries):
        agents = e.get("agents", {})
        for agent_name, agent_info in agents.items():
            # Token 消耗 (使用实际记录的 total_tokens)
            token_data[agent_name][i] = agent_info.get("total_tokens", 0)
            
            # 工具调用次数
            tc_count = 0
            tool_calls_list = e.get("tool_calls", [])
            if isinstance(tool_calls_list, list):
                for tc_entry in tool_calls_list:
                    if isinstance(tc_entry, dict):
                        # v1 格式: {"round": 1, "tool_calls": [{"function": "...", ...}]}
                        calls = tc_entry.get("tool_calls", [])
                        if isinstance(calls, list) and calls:
                            tc_count += len(calls)
                        # v2 格式: {"round": 1, "function": "read_novel_lines", "args": {...}}
                        elif tc_entry.get("function"):
                            tc_count += 1
            
            tool_data[agent_name][i] = tc_count
    
    return sorted_agents, rounds, token_data, tool_data


def plot_token_consumption(agents, rounds, token_data, save_path):
    """绘制每轮每个 agent 的 token 消耗堆叠图"""
    n = len(rounds)
    
    fig, ax = plt.subplots(figsize=(max(16, n * 0.15), 7))
    
    x = np.arange(n)
    bottom = np.zeros(n)
    colors = plt.cm.tab10(np.linspace(0, 1, len(agents)))
    
    for idx, agent in enumerate(agents):
        values = np.array(token_data[agent])
        ax.bar(x, values, bottom=bottom, label=agent, color=colors[idx], width=0.8)
        bottom += values
    
    ax.set_xlabel('轮次', fontsize=12)
    ax.set_ylabel('Token 消耗', fontsize=12)
    ax.set_title('每轮各 Agent Token 消耗', fontsize=14)
    ax.legend(loc='upper right')
    ax.set_xticks(x)
    # 只显示部分标签避免重叠
    step = max(1, n // 30)
    ax.set_xticklabels([str(r) if i % step == 0 else '' for i, r in enumerate(rounds)])
    ax.set_xlim(-0.5, n - 0.5)
    
    # 添加总 token 标注
    total_tokens = sum(sum(v) for v in token_data.values())
    ax.text(0.98, 0.98, f"总 Token: {total_tokens:,}", transform=ax.transAxes,
            va='top', ha='right', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Token 消耗图已保存: {save_path}")


def plot_tool_calls(agents, rounds, tool_data, save_path):
    """绘制每轮每个 agent 的工具调用次数堆叠图"""
    n = len(rounds)
    
    fig, ax = plt.subplots(figsize=(max(16, n * 0.15), 7))
    
    x = np.arange(n)
    bottom = np.zeros(n)
    colors = plt.cm.Set2(np.linspace(0, 1, len(agents)))
    
    for idx, agent in enumerate(agents):
        values = np.array(tool_data[agent])
        ax.bar(x, values, bottom=bottom, label=agent, color=colors[idx], width=0.8)
        bottom += values
    
    ax.set_xlabel('轮次', fontsize=12)
    ax.set_ylabel('工具调用次数', fontsize=12)
    ax.set_title('每轮各 Agent 工具调用次数', fontsize=14)
    ax.legend(loc='upper right')
    ax.set_xticks(x)
    step = max(1, n // 30)
    ax.set_xticklabels([str(r) if i % step == 0 else '' for i, r in enumerate(rounds)])
    ax.set_xlim(-0.5, n - 0.5)
    
    total_calls = sum(sum(v) for v in tool_data.values())
    ax.text(0.98, 0.98, f"总工具调用: {int(total_calls)}", transform=ax.transAxes,
            va='top', ha='right', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  工具调用图已保存: {save_path}")


def print_statistics(agents, rounds, token_data, tool_data):
    """打印统计数据"""
    print("\n" + "=" * 60)
    print("  统计数据")
    print("=" * 60)
    print(f"  总轮次: {len(rounds)}")
    
    print(f"\n  Agent 名称: {', '.join(agents)}")
    
    print(f"\n  --- Token 消耗 ---")
    for agent in agents:
        values = token_data[agent]
        total = sum(values)
        nonzero = [v for v in values if v > 0]
        avg = total / max(len(nonzero), 1)
        print(f"  {agent}: 总计={total:,}, 平均每轮={avg:,.0f}, 有消耗轮次={len(nonzero)}")
    
    print(f"\n  --- 工具调用 ---")
    for agent in agents:
        values = tool_data[agent]
        total = sum(values)
        print(f"  {agent}: 总计={int(total)}, 有调用轮次={sum(1 for v in values if v > 0)}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("  NovelSpeakerV4 日志分析")
    print("=" * 60)
    
    entries = load_logs(LOG_PATH)
    print(f"\n  加载日志: {len(entries)} 条")
    
    agents, rounds, token_data, tool_data = extract_agent_data(entries)
    
    print_statistics(agents, rounds, token_data, tool_data)
    
    # 保存 CSV 数据
    csv_path = os.path.join(OUT_DIR, "agent_stats.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        # Header
        headers = ["round"] + [f"{a}_tokens" for a in agents] + [f"{a}_tool_calls" for a in agents]
        f.write(",".join(headers) + "\n")
        
        for i, r in enumerate(rounds):
            row = [str(r)]
            for a in agents:
                row.append(str(token_data[a][i]))
            for a in agents:
                row.append(str(int(tool_data[a][i])))
            f.write(",".join(row) + "\n")
    print(f"\n  CSV 数据已保存: {csv_path}")
    
    # 图表1: Token 消耗
    plot_token_consumption(agents, rounds, token_data, 
                           os.path.join(OUT_DIR, "token_consumption.png"))
    
    # 图表2: 工具调用
    plot_tool_calls(agents, rounds, tool_data, 
                    os.path.join(OUT_DIR, "tool_calls.png"))
    
    print(f"\n  图表已保存至: {OUT_DIR}")


if __name__ == "__main__":
    main()
