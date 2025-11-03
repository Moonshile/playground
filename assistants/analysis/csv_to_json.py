#!/usr/bin/env python3
"""
csv_to_json.py

功能：
将 CSV 格式的聊天记录转换为按 userId 聚合后的 JSON 文件。
"""

import sys
import json
import csv
from collections import defaultdict, Counter


def clean_field_name(field_name):
    """
    清理字段名，去除 BOM 字符和首尾空白

    Args:
        field_name (str): 原始字段名

    Returns:
        str: 清理后的字段名
    """
    if not field_name:
        return field_name
    # 去除 BOM 字符 (U+FEFF)
    field_name = field_name.replace('\ufeff', '')
    # 去除首尾空白
    return field_name.strip()


def read_csv_chat_records(input_file, encoding='utf-8-sig'):
    """
    读取 CSV 聊天记录文件

    Args:
        input_file (str): 输入 CSV 文件路径
        encoding (str): 文件编码（默认: utf-8-sig，自动处理 BOM）

    Returns:
        list: 聊天记录列表
    """
    records = []
    with open(input_file, 'r', encoding=encoding, newline='') as f:
        reader = csv.DictReader(f)
        # 清理字段名中的 BOM 字符
        original_fieldnames = reader.fieldnames
        if original_fieldnames:
            cleaned_fieldnames = [clean_field_name(fn) for fn in original_fieldnames]
            reader.fieldnames = cleaned_fieldnames

        for row in reader:
            # 清理每条记录中的字段名（如果原始字段名中有 BOM，DictReader 可能已经处理）
            # 但为了保险起见，创建一个新字典，确保字段名干净
            cleaned_row = {}
            for key, value in row.items():
                clean_key = clean_field_name(key)
                cleaned_row[clean_key] = value
            records.append(cleaned_row)
    return records


def aggregate_by_user_id(records):
    """
    按 userId 聚合聊天记录，并按时间顺序排序每个用户的消息

    Args:
        records (list): 聊天记录列表

    Returns:
        dict: 按 userId 聚合的数据，格式为 {userId: [messages]}
    """
    aggregated = defaultdict(list)

    for record in records:
        user_id = record.get('userId', '')
        if user_id:
            aggregated[user_id].append(record)

    # 转换为普通字典，并按时间排序每个用户的消息
    result = {}
    for user_id, messages in aggregated.items():
        # 按 dbctime 排序消息（空值或无效值排到最后）
        def sort_key(x):
            dbctime = x.get('dbctime', '')
            if not dbctime or dbctime == 'null':
                return '9999-99-99 99:99:99'  # 将空值排到最后
            return dbctime

        sorted_messages = sorted(messages, key=sort_key)
        result[user_id] = sorted_messages

    return result


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


def bucket_message_counts(message_counts):
    """
    将消息数量分布合并成桶，便于阅读

    Args:
        message_counts (list): 每个用户的消息数量列表

    Returns:
        dict: 桶化的分布，格式为 {bucket_label: user_count}
    """
    buckets = defaultdict(int)

    for count in message_counts:
        if count <= 1:
            buckets["1条"] += 1
        elif count == 2:
            buckets["2条"] += 1
        elif count <= 5:
            buckets["3-5条"] += 1
        elif count <= 10:
            buckets["6-10条"] += 1
        elif count <= 20:
            buckets["11-20条"] += 1
        elif count <= 50:
            buckets["21-50条"] += 1
        elif count <= 100:
            buckets["51-100条"] += 1
        elif count <= 200:
            buckets["101-200条"] += 1
        elif count <= 500:
            buckets["201-500条"] += 1
        else:
            buckets["500+条"] += 1

    return buckets


def print_statistics(aggregated):
    """
    打印统计信息：用户数、消息总数、消息数量分布

    Args:
        aggregated (dict): 按 userId 聚合的数据
    """
    user_count = len(aggregated)
    total_messages = sum(len(messages) for messages in aggregated.values())

    # 计算消息数量分布
    message_counts = [len(messages) for messages in aggregated.values()]

    print("\n" + "="*60)
    print("📊 统计信息")
    print("="*60)
    print(f"👥 用户总数: {user_count}")
    print(f"💬 消息总数: {total_messages}")
    print(f"\n📈 用户消息数量分布:")

    if message_counts:
        sorted_counts = sorted(message_counts)

        # 计算中位数
        n = len(sorted_counts)
        if n % 2 == 0:
            median = (sorted_counts[n // 2 - 1] + sorted_counts[n // 2]) / 2
        else:
            median = sorted_counts[n // 2]

        print(f"  最少消息数: {min(message_counts)}")
        print(f"  最多消息数: {max(message_counts)}")
        print(f"  平均消息数: {total_messages / user_count:.2f}")
        print(f"  中位数消息数: {median:.2f}")
        print(f"\n  分布统计:")

        # 将消息数量分布合并成桶
        buckets = bucket_message_counts(message_counts)

        # 定义桶的显示顺序
        bucket_order = ["1条", "2条", "3-5条", "6-10条", "11-20条",
                       "21-50条", "51-100条", "101-200条", "201-500条", "500+条"]

        # 按顺序显示，只显示有用户的桶
        for bucket_label in bucket_order:
            if bucket_label in buckets:
                user_num = buckets[bucket_label]
                percentage = (user_num / user_count) * 100
                bar = "█" * int(percentage / 2)  # 简单的文本条形图
                print(f"    {bucket_label:>12}: {user_num:>4} 个用户 ({percentage:>5.1f}%) {bar}")

    print("="*60)


def main():
    if len(sys.argv) < 3:
        print("用法: python csv_to_json.py <input_csv_file> <output_json_file> [encoding]")
        print("示例: python csv_to_json.py chat_records.csv output.json")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    encoding = sys.argv[3] if len(sys.argv) > 3 else 'utf-8-sig'

    try:
        print(f"📖 正在读取 CSV 文件: {input_file}")
        records = read_csv_chat_records(input_file, encoding)
        print(f"✅ 读取到 {len(records)} 条记录")

        print(f"🔄 正在按 userId 聚合数据...")
        aggregated = aggregate_by_user_id(records)
        print(f"✅ 聚合完成")

        # 打印统计信息
        print_statistics(aggregated)

        print(f"\n💾 正在保存到 JSON 文件: {output_file}")
        save_json(output_file, aggregated)
        print(f"✅ 保存完成！")

    except FileNotFoundError:
        print(f"❌ 错误: 文件不存在: {input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

