#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

def extract_transcript_ids(input_file, output_file, encoding='utf-8-sig'):
    """
    从CSV文件中提取id和recordInfo.processedTranscriptResourceId字段。
    
    Args:
        input_file (str): 输入CSV文件路径
        output_file (str): 输出CSV文件路径
        encoding (str): 文件编码（默认: utf-8-sig）
    """
    try:
        # 确保输出目录存在
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        valid_rows = []  # 存储有效的行
        
        # 读取CSV文件
        with open(input_file, 'r', encoding=encoding) as infile:
            reader = csv.DictReader(infile)
            
            # 验证必需的列是否存在
            required_fields = {'id', 'recordInfo'}
            if not required_fields.issubset(reader.fieldnames):
                missing = required_fields - set(reader.fieldnames)
                print(f"Error: 输入文件缺少必需的列: {missing}")
                return
            
            # 处理每一行
            row_count = 0
            valid_count = 0
            for row in reader:
                row_count += 1
                try:
                    # 获取id
                    record_id = row['id']
                    if not record_id:
                        continue
                        
                    # 解析recordInfo JSON
                    record_info = row['recordInfo']
                    if not record_info:
                        continue
                        
                    try:
                        record_info_json = json.loads(record_info)
                    except json.JSONDecodeError:
                        print(f"Warning: 第 {row_count} 行的recordInfo不是有效的JSON")
                        continue
                    
                    # 提取processedTranscriptResourceId
                    transcript_id = record_info_json.get('processedTranscriptResourceId')
                    if not transcript_id:
                        continue
                    
                    # 保存有效的行
                    valid_rows.append([record_id, transcript_id])
                    valid_count += 1
                    
                    # 显示进度
                    if row_count % 1000 == 0:
                        print(f"Progress: 已处理 {row_count} 行，找到 {valid_count} 个有效记录")
                        
                except Exception as e:
                    print(f"Warning: 处理第 {row_count} 行时出错: {str(e)}")
                    continue
            
        # 写入输出文件
        with open(output_file, 'w', newline='', encoding=encoding) as outfile:
            writer = csv.writer(outfile)
            # 写入表头
            writer.writerow(['id', 'transcript_id'])
            # 写入数据
            writer.writerows(valid_rows)
            
        print(f"\n处理完成:")
        print(f"- 总共处理了 {row_count} 行")
        print(f"- 提取出 {valid_count} 个有效记录")
        print(f"- 结果已保存到: {output_file}")
            
    except FileNotFoundError:
        print(f"Error: 找不到输入文件 '{input_file}'")
    except PermissionError:
        print(f"Error: 没有权限访问文件")
    except Exception as e:
        print(f"Error: 发生意外错误: {str(e)}")

def main():
    example_text = '''
示例:
    1. 基本用法:
    $ ./extract_transcript_ids.py input.csv output.csv

    2. 指定GBK编码:
    $ ./extract_transcript_ids.py input.csv output.csv -e gbk

说明:
    - 输入CSV文件必须包含 'id' 和 'recordInfo' 列
    - recordInfo列应包含JSON数据
    - 只有当recordInfo中的processedTranscriptResourceId字段非空时才会输出该行
    - 输出文件包含两列：id 和 transcript_id
    - 默认使用 utf-8-sig 编码（可以处理带BOM的文件）
'''
    
    parser = argparse.ArgumentParser(
        description='从CSV文件中提取id和transcript_id',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=example_text)
    parser.add_argument('input_file', help='输入CSV文件路径')
    parser.add_argument('output_file', help='输出CSV文件路径')
    parser.add_argument('-e', '--encoding', default='utf-8-sig', help='文件编码（默认: utf-8-sig）')
    
    args = parser.parse_args()
    
    # 验证文件是否存在
    if not Path(args.input_file).exists():
        print(f"Error: 输入文件 '{args.input_file}' 不存在")
        return
        
    extract_transcript_ids(args.input_file, args.output_file, args.encoding)

if __name__ == '__main__':
    main()
