#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path
import os

def extract_csv_rows(input_file, output_file, start_row, end_row, delimiter=',', encoding='utf-8'):
    """
    Extract rows from input CSV file and write to output CSV file.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
        start_row (int): Starting row number (0-based, 0 means first row after header)
        end_row (int): Ending row number (exclusive)
        delimiter (str): CSV delimiter character
        encoding (str): File encoding (default: utf-8)
    """
    try:
        # 确保输出目录存在
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 计数器
        total_rows = 0
        processed_rows = 0
        
        # 首先计算总行数
        with open(input_file, 'r', newline='', encoding=encoding) as infile:
            total_rows = sum(1 for _ in infile)
            
        if total_rows == 0:
            print(f"Error: Input file '{input_file}' is empty")
            return
            
        # 验证行号范围
        if start_row < 0:
            print("Error: start_row must be >= 0")
            return
            
        if end_row > total_rows - 1:  # -1 因为不包含表头
            print(f"Warning: end_row ({end_row}) is greater than total data rows ({total_rows-1})")
            end_row = total_rows - 1
        
        if start_row >= end_row:
            print("Error: start_row must be less than end_row")
            return
        
        # 读取并写入数据
        with open(input_file, 'r', newline='', encoding=encoding) as infile, \
             open(output_file, 'w', newline='', encoding=encoding) as outfile:
            
            reader = csv.reader(infile, delimiter=delimiter)
            writer = csv.writer(outfile, delimiter=delimiter)
            
            # 写入表头
            header = next(reader)
            writer.writerow(header)
            processed_rows += 1
            
            # 跳过不需要的行
            for _ in range(start_row):
                next(reader)
                processed_rows += 1
                
            # 写入选定的行
            for _ in range(end_row - start_row):
                try:
                    row = next(reader)
                    writer.writerow(row)
                    processed_rows += 1
                    
                    # 显示进度
                    if processed_rows % 1000 == 0:
                        print(f"Progress: {processed_rows}/{total_rows} rows processed")
                        
                except StopIteration:
                    break
            
        print(f"Successfully extracted rows {start_row} to {end_row-1} to '{output_file}'")
        print(f"Total rows processed: {processed_rows}")
            
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
    except UnicodeDecodeError:
        print(f"Error: Unable to read the file with {encoding} encoding. Try a different encoding.")
    except PermissionError:
        print(f"Error: Permission denied when accessing the files")
    except Exception as e:
        print(f"Error: An unexpected error occurred: {str(e)}")

def main():
    example_text = '''
示例:
    1. 基本用法 - 提取数据行0到99（不包含第99行）:
    $ ./csv_extractor.py input.csv output.csv 0 100

    2. 使用制表符作为分隔符，提取前10行数据（0-9行）:
    $ ./csv_extractor.py input.csv output.csv 0 10 -d $'\\t'

    3. 提取第5行到第14行（共10行）并指定GBK编码:
    $ ./csv_extractor.py input.csv output.csv 5 15 -e gbk

    4. 组合使用 - 从GBK编码的TSV文件中提取20-39行:
    $ ./csv_extractor.py input.tsv output.tsv 20 40 -d $'\\t' -e gbk

注意：
    - 行号从0开始计数（0表示第一个数据行，即表头后的第一行）
    - 结束行号是开区间（例如，end_row=100表示提取到第99行）
    - 输出文件总是包含表头行
'''
    
    parser = argparse.ArgumentParser(
        description='从CSV文件中提取指定范围的数据行（包含表头）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=example_text)
    parser.add_argument('input_file', help='输入CSV文件路径')
    parser.add_argument('output_file', help='输出CSV文件路径')
    parser.add_argument('start_row', type=int, help='起始行号（从0开始，0表示表头后的第一行）')
    parser.add_argument('end_row', type=int, help='结束行号（不包含此行）')
    parser.add_argument('-d', '--delimiter', default=',', help='CSV分隔符（默认: ,）')
    parser.add_argument('-e', '--encoding', default='utf-8', help='文件编码（默认: utf-8）')
    
    args = parser.parse_args()
    
    # 验证文件是否存在
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist")
        return
        
    extract_csv_rows(args.input_file, args.output_file, args.start_row, args.end_row,
                    delimiter=args.delimiter, encoding=args.encoding)

if __name__ == '__main__':
    main()
