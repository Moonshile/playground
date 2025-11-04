#!/usr/bin/env python3
"""
analyze_interactive_games.py

功能：
分析JSON格式的聊天历史记录，识别用户与AI的"互动游戏"行为。
支持GPT和Gemini两种模型进行分析。
"""

import sys
import json
import csv
import os
import time
from typing import List, Dict, Any
from openai import OpenAI
import google.genai as genai

ANALYSIS_PROMPT = """你是一个专业的聊天行为分析师。请分析以下用户与AI的聊天记录，识别其中是否存在"互动游戏"行为。

定义：
- "互动游戏" = 用户主动提出玩法/规则/角色扮演/回合机制/表情或符号约定/情绪或语气模拟/猜谜竞赛等。
- 需要出现"发起或默认接受的规则/玩法"，而不只是闲聊或普通问答。

请仔细分析聊天记录，识别所有的互动游戏。对于每个识别到的游戏，需要提取：
1. 游戏名称
2. 游戏规则
3. 聊天内容示例（展示最能体现该游戏的对话片段，1-3条消息即可）

如果存在多个不同类型的游戏，请分别列出。

请以JSON格式返回结果，格式如下：
{
  "games": [
    {
      "game_name": "游戏名称",
      "game_rules": "游戏规则描述",
      "content_example": "聊天内容示例"
    }
  ]
}

如果没有识别到任何互动游戏，返回：{"games": []}

现在请分析以下聊天记录：

"""


def load_json(input_file, encoding='utf-8'):
    """
    读取 JSON 文件

    Args:
        input_file (str): 输入 JSON 文件路径
        encoding (str): 文件编码（默认: utf-8）

    Returns:
        dict: JSON 数据
    """
    with open(input_file, 'r', encoding=encoding) as f:
        return json.load(f)


def format_chat_history(messages: List[Dict[str, Any]]) -> str:
    """
    将聊天记录格式化为文本

    Args:
        messages: 消息列表

    Returns:
        str: 格式化后的聊天记录文本
    """
    formatted = []
    for msg in messages:
        msg_type = msg.get('type', '')
        content = msg.get('content', '')
        dbctime = msg.get('dbctime', '')

        # type=1 通常是用户消息，type=40 通常是AI回复（根据示例判断）
        role = "用户" if msg_type == "1" else "AI"
        formatted.append(f"[{dbctime}] {role}: {content}")

    return "\n".join(formatted)


def get_chat_time_range(messages: List[Dict[str, Any]]) -> str:
    """
    获取聊天记录的时间范围

    Args:
        messages: 消息列表

    Returns:
        str: 时间范围字符串
    """
    if not messages:
        return ""

    times = [msg.get('dbctime', '') for msg in messages if msg.get('dbctime')]
    if not times:
        return ""

    times = sorted(times)
    start_time = times[0]
    end_time = times[-1]

    if start_time == end_time:
        return start_time
    return f"{start_time} ~ {end_time}"


def call_gpt_model(client: OpenAI, chat_history: str, model: str = "gpt-4o") -> Dict[str, Any]:
    """
    调用GPT模型进行分析

    Args:
        client: OpenAI客户端
        chat_history: 聊天历史文本
        model: 模型名称

    Returns:
        dict: 解析后的JSON结果
    """
    messages = [
        {"role": "system", "content": "你是一个专业的聊天行为分析师。请严格按照JSON格式返回分析结果。"},
        {"role": "user", "content": ANALYSIS_PROMPT + chat_history + "\n\n请只返回JSON格式的结果，不要包含其他文本。"}
    ]

    # 尝试使用 response_format，如果不支持则忽略
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            response_format={"type": "json_object"}
        )
    except Exception:
        # 如果不支持 response_format，则使用普通调用
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3
        )

    result_text = response.choices[0].message.content.strip()

    # 尝试提取JSON（如果返回内容包含其他文本）
    if result_text.startswith("```json"):
        result_text = result_text.replace("```json", "").replace("```", "").strip()
    elif result_text.startswith("```"):
        result_text = result_text.replace("```", "").strip()

    return json.loads(result_text)


def call_gemini_model(chat_history: str, model_name: str = "gemini-2.0-flash-exp", api_key: str = None) -> Dict[str, Any]:
    """
    调用Gemini模型进行分析（使用最新版 google-genai API）

    Args:
        chat_history: 聊天历史文本
        model_name: 模型名称
        api_key: Google API 密钥

    Returns:
        dict: 解析后的JSON结果
    """
    # 使用最新的 API 创建客户端
    client = genai.Client(api_key=api_key) if api_key else genai.Client()
    prompt = ANALYSIS_PROMPT + chat_history + "\n\n请以JSON格式返回结果，只返回JSON，不要包含其他文本。"

    # 使用最新的 API 调用方式
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config={
            "temperature": 0.3,
        }
    )

    # 处理响应
    if hasattr(response, 'text'):
        result_text = response.text.strip()
    elif hasattr(response, 'candidates') and len(response.candidates) > 0:
        # 备用方案：从 candidates 获取
        candidate = response.candidates[0]
        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
            result_text = candidate.content.parts[0].text.strip()
        else:
            raise ValueError("无法从响应中获取文本内容")
    else:
        raise ValueError("无法从响应中获取文本内容")

    # 尝试提取JSON（如果返回内容包含其他文本）
    if result_text.startswith("```json"):
        result_text = result_text.replace("```json", "").replace("```", "").strip()
    elif result_text.startswith("```"):
        result_text = result_text.replace("```", "").strip()

    return json.loads(result_text)


def analyze_user_chat(user_id: str, messages: List[Dict[str, Any]],
                     model_type: str, model_name: str) -> List[Dict[str, Any]]:
    """
    分析单个用户的聊天记录

    Args:
        user_id: 用户ID
        messages: 消息列表
        model_type: 模型类型 ("gpt" 或 "gemini")
        model_name: 具体的模型名称

    Returns:
        list: 识别到的游戏列表，每个游戏是一个字典
    """
    if not messages:
        return []

    chat_history = format_chat_history(messages)

    try:
        if model_type.lower() == "gpt":
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            if not client.api_key:
                raise ValueError("请设置 OPENAI_API_KEY 环境变量")
            result = call_gpt_model(client, chat_history, model_name)
        elif model_type.lower() == "gemini":
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("请设置 GOOGLE_API_KEY 环境变量")
            result = call_gemini_model(chat_history, model_name, api_key)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}，请使用 'gpt' 或 'gemini'")

        games = result.get("games", [])
        if not isinstance(games, list):
            games = []

        return games

    except json.JSONDecodeError as e:
        print(f"  ⚠️  用户 {user_id} 的返回结果JSON解析失败: {str(e)}")
        return []
    except Exception as e:
        print(f"  ⚠️  用户 {user_id} 分析失败: {str(e)}")
        return []


def save_results_to_csv(output_file: str, results: List[Dict[str, Any]], encoding='utf-8', append=False):
    """
    保存结果到CSV文件

    Args:
        output_file: 输出文件路径
        results: 结果列表，每个元素包含：序号、用户ID、聊天记录时间、游戏名称、游戏规则、聊天内容示例、聊天记录
        encoding: 文件编码
        append: 是否追加模式（如果为True，且文件已存在，则追加数据；否则写入表头）
    """
    file_exists = os.path.exists(output_file) and os.path.getsize(output_file) > 0

    mode = 'a' if append and file_exists else 'w'
    with open(output_file, mode, newline='', encoding=encoding) as f:
        writer = csv.writer(f)

        # 如果不是追加模式或文件不存在，写入表头
        if not (append and file_exists):
            writer.writerow(['序号', '用户ID', '聊天记录时间', '游戏名称', '游戏规则', '聊天内容示例', '聊天记录'])

        # 写入数据
        for row in results:
            writer.writerow([
                row['序号'],
                row['用户ID'],
                row['聊天记录时间'],
                row['游戏名称'],
                row['游戏规则'],
                row['聊天内容示例'],
                row.get('聊天记录', '')
            ])


def generate_chat_link(user_id: str) -> str:
    """
    根据用户ID生成聊天记录链接

    Args:
        user_id: 用户ID

    Returns:
        str: 聊天记录链接
    """
    return f"https://assistants.classup.info/tower?userId={user_id}"


def read_existing_csv(output_file: str, encoding='utf-8') -> tuple[int, set[str]]:
    """
    读取已存在的CSV文件，获取当前序号和已处理的用户ID集合

    Args:
        output_file: CSV文件路径
        encoding: 文件编码

    Returns:
        tuple: (当前最大序号, 已处理的用户ID集合)
    """
    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
        return 0, set()

    max_count = 0
    processed_users = set()

    try:
        with open(output_file, 'r', encoding=encoding, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    count = int(row.get('序号', 0))
                    max_count = max(max_count, count)
                    user_id = row.get('用户ID', '')
                    if user_id:
                        processed_users.add(user_id)
                except (ValueError, KeyError):
                    continue
    except Exception as e:
        print(f"⚠️  读取已有CSV文件时出错: {str(e)}")
        return 0, set()

    return max_count, processed_users


def append_result_to_csv(output_file: str, row_data: Dict[str, Any], current_count: int, encoding='utf-8'):
    """
    追加单条结果到CSV文件

    Args:
        output_file: 输出文件路径
        row_data: 单条结果数据
        current_count: 当前序号
        encoding: 文件编码
    """
    file_exists = os.path.exists(output_file) and os.path.getsize(output_file) > 0

    with open(output_file, 'a', newline='', encoding=encoding) as f:
        writer = csv.writer(f)

        # 如果文件不存在，先写入表头
        if not file_exists:
            writer.writerow(['序号', '用户ID', '聊天记录时间', '游戏名称', '游戏规则', '聊天内容示例', '聊天记录'])

        # 写入数据
        writer.writerow([
            current_count,
            row_data['用户ID'],
            row_data['聊天记录时间'],
            row_data['游戏名称'],
            row_data['游戏规则'],
            row_data['聊天内容示例'],
            row_data.get('聊天记录', '')
        ])


def main():
    if len(sys.argv) < 5:
        print("用法: python analyze_interactive_games.py <input_json_file> <output_csv_file> <model_type> <model_name> [min_message_count] [start_user_id] [encoding]")
        print("参数说明:")
        print("  input_json_file: 输入的JSON文件路径")
        print("  output_csv_file: 输出的CSV文件路径")
        print("  model_type: 模型类型 (gpt 或 gemini)")
        print("  model_name: 模型名称 (如 gpt-4o, gemini-2.0-flash-exp 等)")
        print("  min_message_count: 可选，最小消息数量阈值（默认: 0，即分析所有用户）")
        print("  start_user_id: 可选，从指定用户ID开始继续分析（如果文件已存在，会自动继续）")
        print("  encoding: 可选，文件编码（默认: utf-8）")
        print("\n示例:")
        print("  python analyze_interactive_games.py data.json output.csv gpt gpt-4o")
        print("  python analyze_interactive_games.py data.json output.csv gemini gemini-pro 10")
        print("  python analyze_interactive_games.py data.json output.csv gemini gemini-pro 10 845810418")
        print("  python analyze_interactive_games.py data.json output.csv gemini gemini-pro 10 845810418 utf-8")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    model_type = sys.argv[3]
    model_name = sys.argv[4]

    # 解析可选参数
    min_message_count = 0
    start_user_id = None
    encoding = 'utf-8'

    # 参数解析：按位置解析
    if len(sys.argv) > 5:
        arg5 = sys.argv[5]
        try:
            min_message_count = int(arg5)
        except ValueError:
            # 第5个参数不是数字，可能是用户ID或编码
            if arg5.isdigit():
                start_user_id = arg5
            else:
                encoding = arg5

    if len(sys.argv) > 6:
        arg6 = sys.argv[6]
        if start_user_id is None:
            # 如果第5个参数是数字（最小消息数），第6个参数就是用户ID或编码
            if arg6.isdigit():
                start_user_id = arg6
            else:
                encoding = arg6
        else:
            # 如果第5个参数已经是用户ID，第6个参数就是编码
            encoding = arg6

    if len(sys.argv) > 7:
        encoding = sys.argv[7]

    if model_type.lower() not in ['gpt', 'gemini']:
        print(f"❌ 错误: 不支持的模型类型 '{model_type}'，请使用 'gpt' 或 'gemini'")
        sys.exit(1)

    try:
        print(f"📖 正在读取 JSON 文件: {input_file}")
        data = load_json(input_file, encoding)
        total_users = len(data)
        print(f"✅ 读取完成，共 {total_users} 个用户")

        # 根据最小消息数量过滤用户
        if min_message_count > 0:
            filtered_data = {uid: msgs for uid, msgs in data.items() if len(msgs) >= min_message_count}
            filtered_count = len(filtered_data)
            print(f"🔍 过滤条件: 消息数量 >= {min_message_count}")
            print(f"📊 过滤后用户数: {filtered_count} (保留 {filtered_count/total_users*100:.1f}%)")
            data = filtered_data
        else:
            print(f"📊 将分析所有 {total_users} 个用户")

        if not data:
            print("❌ 没有符合条件的用户需要分析")
            sys.exit(0)

        print(f"🤖 使用模型: {model_type} ({model_name})")

        # 检查输出文件是否已存在，读取已有数据
        existing_count, processed_users = read_existing_csv(output_file, encoding)
        resume_mode = False

        if os.path.exists(output_file) and existing_count > 0:
            print(f"📄 检测到已有输出文件，当前序号: {existing_count}，已处理用户数: {len(processed_users)}")
            if start_user_id:
                print(f"🎯 将从用户ID '{start_user_id}' 开始继续分析")
                resume_mode = True
            elif len(processed_users) > 0:
                print(f"🔄 将跳过已处理的用户，继续追加结果")
                resume_mode = True

        # 如果指定了起始用户ID，从该用户开始
        if start_user_id and start_user_id not in data:
            print(f"❌ 错误: 指定的起始用户ID '{start_user_id}' 不存在于数据中")
            sys.exit(1)

        # 过滤需要处理的用户
        users_to_process = []
        skip_count = 0
        start_idx = None

        for idx, (user_id, messages) in enumerate(data.items(), 1):
            # 如果指定了起始用户ID，找到该用户的位置
            if start_user_id and user_id == start_user_id:
                start_idx = idx
                users_to_process.append((idx, user_id, messages))
            elif start_user_id and start_idx is None:
                continue  # 还没找到起始用户，继续跳过
            elif start_idx is not None:
                # 已经过了起始用户，加入处理列表
                users_to_process.append((idx, user_id, messages))
            elif start_user_id is None and resume_mode:
                # 如果没有指定起始用户但文件存在，跳过已处理的用户
                if user_id in processed_users:
                    skip_count += 1
                    continue
                users_to_process.append((idx, user_id, messages))
            elif start_user_id is None and not resume_mode:
                # 全新开始
                users_to_process.append((idx, user_id, messages))

        if skip_count > 0:
            print(f"⏭️  跳过已处理的用户: {skip_count} 个")

        if not users_to_process:
            print("✅ 所有用户都已处理完成，无需继续分析")
            sys.exit(0)

        print(f"📊 待处理用户数: {len(users_to_process)}")
        print(f"\n🔄 开始分析...\n")

        total_results = existing_count
        processed = 0
        start_time = time.time()
        user_count_with_games = set(processed_users)  # 包含已处理的用户

        for idx, user_id, messages in users_to_process:
            print(f"[{idx}/{len(data)}] 正在分析用户 {user_id} (消息数: {len(messages)})...", end=" ")

            chat_time = get_chat_time_range(messages)
            games = analyze_user_chat(user_id, messages, model_type, model_name)

            if games:
                print(f"✅ 识别到 {len(games)} 个游戏")
                user_count_with_games.add(user_id)

                # 生成聊天记录链接
                chat_link = generate_chat_link(user_id)

                # 每识别到一个游戏就立即保存
                for game in games:
                    total_results += 1
                    row_data = {
                        '用户ID': user_id,
                        '聊天记录时间': chat_time,
                        '游戏名称': game.get('game_name', ''),
                        '游戏规则': game.get('game_rules', ''),
                        '聊天内容示例': game.get('content_example', ''),
                        '聊天记录': chat_link
                    }
                    append_result_to_csv(output_file, row_data, total_results, encoding)
            else:
                print("未识别到游戏")

            processed += 1

            # 显示进度
            if processed % 10 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / processed if processed > 0 else 0
                remaining_users = len(users_to_process) - processed
                remaining = remaining_users * avg_time
                print(f"\n   进度: {processed}/{len(users_to_process)} ({processed/len(users_to_process)*100:.1f}%), "
                      f"预计剩余时间: {remaining/60:.1f} 分钟, "
                      f"已识别游戏数: {total_results}\n")

            # 避免API调用过于频繁
            time.sleep(0.5)

        elapsed = time.time() - start_time
        print(f"\n✅ 分析完成！")
        print(f"   总耗时: {elapsed/60:.1f} 分钟")
        print(f"   已分析用户数: {processed}")
        print(f"   识别到游戏数量: {total_results}")
        print(f"   涉及用户数: {len(user_count_with_games)}")
        print(f"   结果已保存到: {output_file}")

    except FileNotFoundError:
        print(f"❌ 错误: 文件不存在: {input_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ 错误: JSON 解析失败: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

