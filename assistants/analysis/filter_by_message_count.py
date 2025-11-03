#!/usr/bin/env python3
"""
filter_by_message_count.py

功能：
根据消息数量阈值过滤 JSON 文件中的用户数据。
只保留消息数量不低于指定阈值的用户。
"""

import sys
import json
import os


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


def filter_by_message_count(data, min_count):
    """
    根据消息数量阈值过滤数据

    Args:
        data (dict): 按 userId 聚合的数据
        min_count (int): 最小消息数量阈值

    Returns:
        dict: 过滤后的数据
    """
    filtered = {}
    for user_id, messages in data.items():
        if len(messages) >= min_count:
            filtered[user_id] = messages
    return filtered


def save_json(output_file, data, encoding='utf-8', indent=2):
    """
    保存数据为 JSON 文件

    Args:
        output_file (str): 输出 JSON 文件路径
        data (dict): 要保存的数据
        encoding (str): 文件编码（默认: utf-8）
        indent (int): JSON 缩进（默认: 2）
    """
    with open(output_file, 'w', encoding=encoding) as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def print_statistics(original_data, filtered_data, min_count):
    """
    打印过滤统计信息

    Args:
        original_data (dict): 原始数据
        filtered_data (dict): 过滤后的数据
        min_count (int): 最小消息数量阈值
    """
    original_user_count = len(original_data)
    original_message_count = sum(len(msgs) for msgs in original_data.values())

    filtered_user_count = len(filtered_data)
    filtered_message_count = sum(len(msgs) for msgs in filtered_data.values())

    print("\n" + "="*60)
    print("📊 过滤统计信息")
    print("="*60)
    print(f"📋 过滤条件: 消息数量 >= {min_count}")
    print(f"\n原始数据:")
    print(f"  用户数: {original_user_count}")
    print(f"  消息数: {original_message_count}")
    print(f"\n过滤后数据:")
    print(f"  用户数: {filtered_user_count} ({filtered_user_count/original_user_count*100:.1f}%)")
    print(f"  消息数: {filtered_message_count} ({filtered_message_count/original_message_count*100:.1f}%)")
    print(f"  过滤掉: {original_user_count - filtered_user_count} 个用户")
    print("="*60)


def main():
    if len(sys.argv) < 4:
        print("用法: python filter_by_message_count.py <input_json_file> <output_json_file> <min_message_count> [encoding]")
        print("示例: python filter_by_message_count.py data.json filtered.json 10")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    try:
        min_count = int(sys.argv[3])
        if min_count < 0:
            print("❌ 错误: 最小消息数量必须 >= 0")
            sys.exit(1)
    except ValueError:
        print("❌ 错误: 最小消息数量必须是整数")
        sys.exit(1)

    encoding = sys.argv[4] if len(sys.argv) > 4 else 'utf-8'

    try:
        print(f"📖 正在读取 JSON 文件: {input_file}")
        original_data = load_json(input_file, encoding)
        print(f"✅ 读取完成，共 {len(original_data)} 个用户")

        print(f"🔍 正在过滤消息数量 >= {min_count} 的用户...")
        filtered_data = filter_by_message_count(original_data, min_count)
        print(f"✅ 过滤完成，保留 {len(filtered_data)} 个用户")

        # 打印统计信息
        print_statistics(original_data, filtered_data, min_count)

        print(f"\n💾 正在保存到 JSON 文件: {output_file}")
        save_json(output_file, filtered_data, encoding)
        print(f"✅ 保存完成！")

    except FileNotFoundError:
        print(f"❌ 错误: 文件不存在: {input_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ 错误: JSON 解析失败: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

