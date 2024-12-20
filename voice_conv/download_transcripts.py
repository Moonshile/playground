#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin

import requests

# OSS基础URL
OSS_BASE_URL = "https://epoch-ue-prod.oss-us-east-1.aliyuncs.com/"

def download_transcript(transcript_id, output_dir, session):
    """
    下载单个转写文件
    
    Args:
        transcript_id: OSS中的文件路径
        output_dir: 本地保存目录
        session: requests会话对象
    
    Returns:
        tuple: (transcript_id, success, message)
    """
    try:
        # 构建完整URL
        url = urljoin(OSS_BASE_URL, transcript_id)
        
        # 只使用文件名作为保存路径
        filename = os.path.basename(transcript_id)
        local_path = Path(output_dir) / filename
        
        # 下载文件
        response = session.get(url)
        response.raise_for_status()
        
        # 保存文件
        with open(local_path, 'wb') as f:
            f.write(response.content)
            
        return transcript_id, True, "Success"
        
    except requests.exceptions.RequestException as e:
        return transcript_id, False, f"Download failed: {str(e)}"
    except Exception as e:
        return transcript_id, False, f"Error: {str(e)}"

def download_transcripts(input_file, output_dir, max_workers=5, encoding='utf-8-sig'):
    """
    从CSV文件中读取转写ID并下载对应的文件
    
    Args:
        input_file: 包含transcript_id的CSV文件
        output_dir: 下载文件保存目录
        max_workers: 最大并发下载数
        encoding: CSV文件编码
    """
    try:
        # 确保输出目录存在
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 读取CSV文件
        with open(input_file, 'r', encoding=encoding) as f:
            reader = csv.DictReader(f)
            
            # 验证必需的列
            if 'transcript_id' not in reader.fieldnames:
                print("Error: CSV文件必须包含'transcript_id'列")
                return
                
            # 收集所有需要下载的文件
            download_list = [(row['transcript_id'], output_dir) 
                           for row in reader if row['transcript_id']]
        
        total_files = len(download_list)
        if total_files == 0:
            print("没有找到需要下载的文件")
            return
            
        print(f"开始下载 {total_files} 个文件...")
        
        # 创建计数器
        success_count = 0
        fail_count = 0
        
        # 使用会话对象复用连接
        with requests.Session() as session:
            # 设置超时和重试
            session.mount('https://', requests.adapters.HTTPAdapter(
                max_retries=3,
                pool_connections=max_workers,
                pool_maxsize=max_workers
            ))
            
            # 并发下载
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有下载任务
                future_to_id = {
                    executor.submit(download_transcript, tid, out_dir, session): tid
                    for tid, out_dir in download_list
                }
                
                # 处理完成的任务
                for i, future in enumerate(as_completed(future_to_id), 1):
                    transcript_id = future_to_id[future]
                    try:
                        _, success, message = future.result()
                        if success:
                            success_count += 1
                        else:
                            fail_count += 1
                            print(f"Failed to download {transcript_id}: {message}")
                    except Exception as e:
                        fail_count += 1
                        print(f"Error processing {transcript_id}: {str(e)}")
                    
                    # 显示进度
                    if i % 10 == 0 or i == total_files:
                        print(f"Progress: {i}/{total_files} "
                              f"(Success: {success_count}, Failed: {fail_count})")
        
        # 打印最终统计
        print("\n下载完成:")
        print(f"- 总文件数: {total_files}")
        print(f"- 成功: {success_count}")
        print(f"- 失败: {fail_count}")
        print(f"- 文件保存在: {output_dir}")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")

def main():
    example_text = '''
示例:
    1. 基本用法:
    $ ./download_transcripts.py input_transcripts.csv output_dir

    2. 指定并发数和编码:
    $ ./download_transcripts.py input_transcripts.csv output_dir -w 10 -e gbk

说明:
    - 输入CSV文件必须包含 'transcript_id' 列
    - transcript_id应该是OSS中的文件路径
    - 所有文件将直接保存在输出目录下（只使用文件名）
    - 默认使用5个并发下载线程
'''
    
    parser = argparse.ArgumentParser(
        description='从OSS下载转写文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=example_text)
    parser.add_argument('input_file', help='输入CSV文件路径')
    parser.add_argument('output_dir', help='输出目录')
    parser.add_argument('-w', '--workers', type=int, default=5,
                      help='最大并发下载数（默认: 5）')
    parser.add_argument('-e', '--encoding', default='utf-8-sig',
                      help='CSV文件编码（默认: utf-8-sig）')
    
    args = parser.parse_args()
    
    # 验证文件是否存在
    if not Path(args.input_file).exists():
        print(f"Error: 输入文件 '{args.input_file}' 不存在")
        return
        
    download_transcripts(args.input_file, args.output_dir,
                        args.workers, args.encoding)

if __name__ == '__main__':
    main()
