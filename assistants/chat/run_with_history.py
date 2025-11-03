#!/usr/bin/env python3
"""
raku_infer.py

功能：
1️⃣ 从命令行读取输入文件（包含提示词 + 对话历史）
2️⃣ 自动解析：
    - 第一部分 → system message
    - 其余带 user:/assistant: 前缀的内容 → 对应 role
3️⃣ 调用 OpenAI GPT-4.1 模型生成响应
4️⃣ 打印结果并保存为 <filename>_response.md
"""

import sys
import os
import re
import json
from openai import OpenAI

# ---------- 解析函数 ----------

ROLE_PATTERN = re.compile(r"^(assistant|user)\s*:\s*(.*)$", re.IGNORECASE)
SEPARATOR_PATTERN = re.compile(r"^-{8,}\s*$")  # 8个以上短横线视为分隔符


def split_header_and_body(text: str):
    """将文件分为两部分：header（system prompt）和 body（对话部分）"""
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if ROLE_PATTERN.match(line) or SEPARATOR_PATTERN.match(line):
            header = "".join(lines[:i])
            body = "".join(lines[i:])
            return header.strip(), body
    return text.strip(), ""


def parse_role_blocks(body: str):
    """解析 user/assistant 块"""
    if not body.strip():
        return []
    lines = body.splitlines()
    messages = []
    current_role = None
    current_lines = []

    def flush():
        nonlocal current_role, current_lines
        if current_role:
            messages.append({
                "role": current_role,
                "content": "\n".join(current_lines).strip()
            })
        current_role = None
        current_lines = []

    for line in lines:
        m = ROLE_PATTERN.match(line)
        if m:
            flush()
            current_role = m.group(1).lower()
            rest = m.group(2) or ""
            current_lines = [rest]
        else:
            if current_role:
                current_lines.append(line)
            elif messages:
                messages[-1]["content"] += "\n" + line
            else:
                current_role = "user"
                current_lines = [line]
    flush()
    return messages


def build_messages(header, parsed):
    messages = []
    if header:
        messages.append({"role": "system", "content": header})
    messages.extend(parsed)
    return messages


# ---------- 主逻辑 ----------

def main():
    if len(sys.argv) < 2:
        print("用法: python raku_infer.py <conversation_file>")
        sys.exit(1)

    input_path = sys.argv[1]
    if not os.path.exists(input_path):
        print(f"❌ 文件不存在: {input_path}")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    header, body = split_header_and_body(text)
    parsed = parse_role_blocks(body)
    messages = build_messages(header, parsed)

    print("🧩 已解析输入文件:")
    print(f"  System message 字数: {len(header)}")
    print(f"  对话条数: {len(parsed)}")
    print(f"  总消息条数: {len(messages)}")
    print()

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not client:
        print("❌ 请先设置 OPENAI_API_KEY 环境变量")
        sys.exit(1)

    print("🚀 正在调用 GPT-4.1，请稍候...\n")

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        temperature=0.3
    )

    reply = response.choices[0].message.content.strip()

    print("✅ GPT-4.1 响应结果:\n")
    print(reply)

    output_path = os.path.splitext(input_path)[0] + "_response.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(reply)
    print(f"\n💾 已保存结果到: {output_path}")


if __name__ == "__main__":
    main()
