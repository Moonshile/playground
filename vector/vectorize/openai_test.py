"""
使用 OpenAI Embeddings API 将文本转换为向量
支持标准 API 和 Batch API，包含性能统计和费用计算

使用方法:
    python vector/vectorize/openai_test.py -i input.json -o output.json
    python vector/vectorize/openai_test.py -i input.json -o output.json -m text-embedding-3-large -b 50 -r 10
"""
import os
import json
import time
import argparse
from typing import List, Optional, Dict, Any, Tuple
from openai import OpenAI

# ==================== 配置 ====================

# OpenAI Embeddings 模型定价（每百万tokens，美元）
EMBEDDING_PRICING = {
    "text-embedding-3-small": 0.02,  # $0.02 per 1M tokens
    "text-embedding-3-large": 0.13,  # $0.13 per 1M tokens
    "text-embedding-ada-002": 0.10,  # $0.10 per 1M tokens (旧模型)
}


# ==================== 工具函数 ====================

def get_openai_client() -> OpenAI:
    """获取 OpenAI 客户端，从环境变量读取 API key"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("请设置 OPENAI_API_KEY 环境变量")
    return OpenAI(api_key=api_key)


def get_model_pricing(model: str) -> float:
    """获取模型的定价（每百万tokens，美元）"""
    return EMBEDDING_PRICING.get(model, 0.02)  # 默认使用text-embedding-3-small的定价


def calculate_cost(tokens: int, model: str) -> float:
    """
    计算费用（美元）

    Args:
        tokens: token数量
        model: 使用的模型

    Returns:
        费用（美元）
    """
    price_per_million = get_model_pricing(model)
    return (tokens / 1_000_000) * price_per_million


# ==================== 向量化函数 ====================

def vectorize_text(
    text: str,
    model: str = "text-embedding-3-small",
    client: Optional[OpenAI] = None
) -> List[float]:
    """
    使用 OpenAI Embeddings API 将单个文本转换为向量

    Args:
        text: 要转换的文本
        model: 使用的模型，默认为 "text-embedding-3-small"
        client: OpenAI 客户端，如果为 None 则从环境变量创建

    Returns:
        向量列表（浮点数列表）
    """
    if client is None:
        client = get_openai_client()

    if not text or not text.strip():
        raise ValueError("文本不能为空")

    response = client.embeddings.create(model=model, input=text)
    return response.data[0].embedding


def vectorize_texts_batch(
    texts: List[str],
    model: str = "text-embedding-3-small",
    client: Optional[OpenAI] = None,
    batch_size: int = 100
) -> Tuple[List[List[float]], float, Dict[str, int]]:
    """
    使用 OpenAI Embeddings API 批量将文本转换为向量

    Args:
        texts: 要转换的文本列表
        model: 使用的模型，默认为 "text-embedding-3-small"
        client: OpenAI 客户端，如果为 None 则从环境变量创建
        batch_size: 每次请求的批量大小（OpenAI API 支持最多 2048 个输入）

    Returns:
        (向量列表, API调用总时间（秒）, token使用信息字典)
        token使用信息包含: prompt_tokens, total_tokens
    """
    if client is None:
        client = get_openai_client()

    if not texts:
        return [], 0.0, {"prompt_tokens": 0, "total_tokens": 0}

    # 过滤空文本
    valid_texts = [(i, text) for i, text in enumerate(texts) if text and text.strip()]
    if not valid_texts:
        raise ValueError("没有有效的文本")

    all_vectors = [None] * len(texts)
    total_api_time = 0.0
    total_prompt_tokens = 0
    total_tokens = 0

    # 分批处理
    for i in range(0, len(valid_texts), batch_size):
        batch = valid_texts[i:i + batch_size]
        batch_texts = [text for _, text in batch]
        batch_indices = [idx for idx, _ in batch]

        # 只记录API调用时间
        api_start = time.time()
        response = client.embeddings.create(model=model, input=batch_texts)
        api_end = time.time()
        api_time = api_end - api_start
        total_api_time += api_time

        # 提取token使用信息
        if hasattr(response, 'usage') and response.usage:
            total_prompt_tokens += response.usage.prompt_tokens
            total_tokens += response.usage.total_tokens

        # 将结果放回原位置
        for idx, embedding in zip(batch_indices, response.data):
            all_vectors[idx] = embedding.embedding

    return all_vectors, total_api_time, {
        "prompt_tokens": total_prompt_tokens,
        "total_tokens": total_tokens
    }


# ==================== 数据加载和检查点 ====================

def load_qa_data(input_file: str) -> List[Dict[str, Any]]:
    """加载QA数据"""
    print(f"📖 正在加载数据: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✅ 已加载 {len(data)} 条数据")
    return data


def get_default_cumulative_stats() -> Dict[str, Any]:
    """获取默认的累计统计信息"""
    return {
        "query_prompt_tokens": 0,
        "query_total_tokens": 0,
        "document_prompt_tokens": 0,
        "document_total_tokens": 0,
        "total_tokens": 0,
        "total_cost": 0.0
    }


def load_checkpoint(output_file: str) -> Dict[str, Any]:
    """加载检查点（已处理的数据，包含性能和token统计）"""
    checkpoint_file = output_file.replace('.json', '_checkpoint.json')
    if os.path.exists(checkpoint_file):
        print(f"📂 发现检查点文件: {checkpoint_file}")
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            processed_count = checkpoint.get('processed_count', 0)
            print(f"   已处理: {processed_count} 条")

            # 恢复累计的token统计
            cumulative_stats = checkpoint.get('cumulative_stats', get_default_cumulative_stats())
            if cumulative_stats.get("total_tokens", 0) > 0:
                print(f"   累计Token: {cumulative_stats['total_tokens']:,}")
                print(f"   累计费用: ${cumulative_stats.get('total_cost', 0):.4f}")

            return checkpoint
        except json.JSONDecodeError as e:
            print(f"   ⚠️  检查点文件格式错误（可能保存时被中断）: {e}")
            print(f"   💡 建议：检查点文件可能已损坏，将从头开始处理")
            # 备份损坏的检查点文件
            backup_file = checkpoint_file + '.corrupted'
            try:
                import shutil
                shutil.move(checkpoint_file, backup_file)
                print(f"   📦 已备份损坏的检查点文件到: {backup_file}")
            except Exception as backup_error:
                print(f"   ⚠️  无法备份损坏的检查点文件: {backup_error}")
            # 返回空的检查点
            return {
                "processed_count": 0,
                "results": [],
                "performance": [],
                "cumulative_stats": get_default_cumulative_stats()
            }
        except Exception as e:
            print(f"   ❌ 加载检查点文件时出错: {e}")
            print(f"   💡 将从头开始处理")
            return {
                "processed_count": 0,
                "results": [],
                "performance": [],
                "cumulative_stats": get_default_cumulative_stats()
            }

    return {
        "processed_count": 0,
        "results": [],
        "performance": [],
        "cumulative_stats": get_default_cumulative_stats()
    }


def save_checkpoint(output_file: str, checkpoint: Dict[str, Any], verbose: bool = False):
    """
    保存检查点（使用原子性写入，避免文件损坏）

    使用临时文件 + 重命名的方式，确保写入过程的原子性：
    1. 先写入临时文件
    2. 写入成功后再重命名替换原文件
    这样即使写入过程中被中断，原文件也不会被损坏
    """
    checkpoint_file = output_file.replace('.json', '_checkpoint.json')
    temp_file = checkpoint_file + '.tmp'

    try:
        # 先写入临时文件
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)

        # 写入成功后再重命名（原子操作）
        # 在大多数文件系统上，重命名是原子操作
        os.replace(temp_file, checkpoint_file)

        if verbose:
            processed = checkpoint.get('processed_count', 0)
            print(f"   💾 检查点已保存: {processed} 条已处理")
    except Exception as e:
        # 如果写入失败，清理临时文件
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass
        # 重新抛出异常，让调用者知道保存失败
        raise RuntimeError(f"保存检查点失败: {e}") from e


def update_cumulative_stats(
    cumulative_stats: Dict[str, Any],
    token_info: Dict[str, int],
    stats_type: str,
    model: str
):
    """更新累计统计信息"""
    cumulative_stats[f"{stats_type}_prompt_tokens"] += token_info.get("prompt_tokens", 0)
    cumulative_stats[f"{stats_type}_total_tokens"] += token_info.get("total_tokens", 0)
    cumulative_stats["total_tokens"] = (
        cumulative_stats["query_total_tokens"] +
        cumulative_stats["document_total_tokens"]
    )
    cumulative_stats["total_cost"] = calculate_cost(cumulative_stats["total_tokens"], model)


# ==================== 性能报告 ====================

def print_performance_report(
    performance_log: List[Dict[str, Any]],
    processed_count: int,
    total_count: int,
    report_type: str,
    model: str,
    cumulative_stats: Optional[Dict[str, Any]] = None
):
    """打印性能报告（包含token统计和费用）"""
    if not performance_log:
        return

    # 筛选指定类型的性能数据
    type_logs = [log for log in performance_log if log.get("type") == report_type]
    if not type_logs:
        return

    # 计算统计信息
    total_api_time = sum(log.get("time_seconds", 0) for log in type_logs)
    total_items = sum(log.get("batch_size", 0) for log in type_logs)
    total_prompt_tokens = sum(log.get("prompt_tokens", 0) for log in type_logs)
    total_tokens = sum(log.get("total_tokens", 0) for log in type_logs)
    avg_time_per_batch = total_api_time / len(type_logs) if type_logs else 0
    avg_items_per_second = total_items / total_api_time if total_api_time > 0 else 0
    avg_tokens_per_item = total_tokens / total_items if total_items > 0 else 0

    # 计算费用
    cost = calculate_cost(total_tokens, model)

    # 累计统计（如果提供）
    cumulative_tokens = cumulative_stats.get("total_tokens", 0) if cumulative_stats else 0
    cumulative_cost = cumulative_stats.get("total_cost", 0.0) if cumulative_stats else 0.0

    print(f"\n📊 {report_type.upper()} 性能报告:")
    print(f"   已处理批次: {len(type_logs)}")
    print(f"   总处理项数: {total_items}")
    print(f"   总API调用时间: {total_api_time:.2f}s")
    print(f"   平均每批次时间: {avg_time_per_batch:.2f}s")
    print(f"   平均处理速度: {avg_items_per_second:.2f} 项/秒")
    print(f"   Token统计:")
    print(f"     - 总输入Token: {total_prompt_tokens:,}")
    print(f"     - 总Token: {total_tokens:,}")
    print(f"     - 平均每项Token: {avg_tokens_per_item:.1f}")
    print(f"   费用统计:")
    print(f"     - 本次费用: ${cost:.4f}")
    if cumulative_stats:
        print(f"     - 累计Token: {cumulative_tokens:,}")
        print(f"     - 累计费用: ${cumulative_cost:.4f}")
    print(f"   总进度: {processed_count}/{total_count} ({processed_count/total_count*100:.1f}%)")
    print()


# ==================== 主处理函数 ====================

def process_batch(
    texts: List[str],
    text_type: str,
    batch_index: int,
    model: str,
    client: OpenAI,
    batch_size: int,
    processed_count: int,
    total_count: int,
    performance_log: List[Dict[str, Any]],
    cumulative_stats: Dict[str, Any],
    report_interval: int
) -> Tuple[List[List[float]], float, int, int, bool]:
    """
    处理一批文本，返回向量、API时间、token统计

    Returns:
        (向量列表, API总时间, prompt_tokens, total_tokens, should_report)
        should_report: 是否需要打印性能报告
    """
    # API调用前提示（让用户知道程序还在运行）
    if batch_index == 0 or (batch_index + 1) % 10 == 0:
        print(f"   ⏳ 正在调用API处理 {text_type}批次 {batch_index + 1}...")

    batch_vectors, api_time, token_info = vectorize_texts_batch(
        texts, model=model, client=client, batch_size=batch_size
    )

    # 记录性能
    batch_perf = {
        "type": text_type,
        "batch_index": batch_index,
        "batch_size": len(texts),
        "time_seconds": round(api_time, 2),
        "items_per_second": round(len(texts) / api_time, 2) if api_time > 0 else 0,
        "prompt_tokens": token_info.get("prompt_tokens", 0),
        "total_tokens": token_info.get("total_tokens", 0)
    }
    performance_log.append(batch_perf)

    # 更新累计统计
    update_cumulative_stats(cumulative_stats, token_info, text_type, model)

    # 打印进度
    processed = processed_count + (batch_index + 1) * batch_size
    if processed > total_count:
        processed = total_count
    progress = processed / total_count * 100
    print(f"   {text_type.capitalize()}批次 {batch_index + 1}: {len(texts)} 条, "
          f"API调用时间 {api_time:.2f}s, "
          f"Token: {token_info.get('total_tokens', 0):,}, "
          f"总进度: {processed}/{total_count} ({progress:.1f}%)")

    # 返回是否需要打印报告（由调用者决定是否打印和保存检查点）
    should_report = (batch_index + 1) % report_interval == 0

    return (
        batch_vectors,
        api_time,
        token_info.get("prompt_tokens", 0),
        token_info.get("total_tokens", 0),
        should_report
    )


def save_checkpoint_with_results(
    output_file: str,
    checkpoint: Dict[str, Any],
    results: List[Dict[str, Any]],
    processed_count: int,
    performance_log: List[Dict[str, Any]],
    cumulative_stats: Dict[str, Any],
    verbose: bool = False
):
    """
    保存检查点（包含结果）
    注意：无论是否从头开始，运行过程中都会保存检查点，以便中断后可以恢复
    """
    checkpoint["results"] = results
    checkpoint["processed_count"] = processed_count
    checkpoint["performance"] = performance_log
    checkpoint["cumulative_stats"] = cumulative_stats
    save_checkpoint(output_file, checkpoint, verbose=verbose)


def process_qa_data(
    input_file: str,
    output_file: str,
    model: str = "text-embedding-3-small",
    batch_size: int = 100,
    from_scratch: bool = False,
    report_interval: int = 5
):
    """
    处理QA数据，为query和document生成向量

    Args:
        input_file: 输入JSON文件路径
        output_file: 输出JSON文件路径
        model: 使用的模型
        batch_size: 批量处理大小
        from_scratch: 是否从头开始（忽略已有检查点），但运行过程中仍会保存检查点
        report_interval: 性能报告打印间隔（每N个批次）
    """
    print("=" * 60)
    print("处理QA数据 - 生成向量")
    print("=" * 60)
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"模型: {model}")
    print(f"批量大小: {batch_size}")
    if from_scratch:
        print(f"模式: 从头开始（忽略已有检查点）")
    else:
        print(f"模式: 自动恢复（如果存在检查点）")
    print(f"性能报告间隔: 每 {report_interval} 个批次")
    print("=" * 60)

    # 加载数据
    data = load_qa_data(input_file)

    # 加载检查点（如果 from_scratch=True，则忽略已有检查点，从头开始）
    if from_scratch:
        checkpoint = {
            "processed_count": 0,
            "results": [],
            "performance": [],
            "cumulative_stats": get_default_cumulative_stats()
        }
        # 如果存在旧的检查点文件，提示用户
        checkpoint_file = output_file.replace('.json', '_checkpoint.json')
        if os.path.exists(checkpoint_file):
            print(f"⚠️  发现已有检查点文件，但将从头开始处理（检查点文件不会被删除）")
    else:
        checkpoint = load_checkpoint(output_file)
    processed_count = checkpoint["processed_count"]
    results = checkpoint.get("results", [])
    performance_log = checkpoint.get("performance", [])
    cumulative_stats = checkpoint.get("cumulative_stats", get_default_cumulative_stats())

    if processed_count > 0:
        print(f"🔄 从第 {processed_count + 1} 条开始继续处理...")
        if cumulative_stats.get("total_tokens", 0) > 0:
            print(f"   累计Token: {cumulative_stats['total_tokens']:,}")
            print(f"   累计费用: ${cumulative_stats.get('total_cost', 0):.4f}")

    # 获取客户端
    client = get_openai_client()

    # 处理剩余的数据
    remaining_data = data[processed_count:]
    total_remaining = len(remaining_data)

    if total_remaining == 0:
        print("✅ 所有数据已处理完成！")
        return

    print(f"\n📊 处理进度: {processed_count}/{len(data)} ({processed_count/len(data)*100:.1f}%)")
    print(f"   剩余: {total_remaining} 条\n")

    # 提取query和document文本
    queries = [item["query"] for item in remaining_data]
    documents = [item["document"] for item in remaining_data]

    # 处理query向量
    print("🔍 正在处理query向量...")
    query_api_time_total = 0.0
    query_vectors = []
    query_total_prompt_tokens = 0
    query_total_tokens = 0

    num_batches = (len(queries) + batch_size - 1) // batch_size
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(queries))
        batch_queries = queries[start_idx:end_idx]

        try:
            batch_vectors, api_time, prompt_tokens, total_tokens, should_report = process_batch(
                batch_queries, "query", i, model, client, batch_size,
                processed_count, len(data), performance_log, cumulative_stats, report_interval
            )

            query_vectors.extend(batch_vectors)
            query_api_time_total += api_time
            query_total_prompt_tokens += prompt_tokens
            query_total_tokens += total_tokens

            # 更新结果（只添加新处理的数据）
            current_processed = processed_count + end_idx
            for j, qv in enumerate(batch_vectors):
                idx = processed_count + start_idx + j
                if idx < len(results):
                    # 更新已有结果
                    results[idx]["query_vector"] = qv
                else:
                    # 添加新结果
                    results.append({
                        "query": queries[start_idx + j],
                        "document": documents[start_idx + j],
                        "query_vector": qv,
                        "document_vector": None,
                        "score": remaining_data[start_idx + j].get("score")
                    })

            # 在打印性能报告时保存检查点（逻辑分离：先打印报告，再保存检查点）
            if should_report:
                # 打印性能报告
                print_performance_report(
                    performance_log, current_processed, len(data), "query", model, cumulative_stats
                )
                # 保存检查点
                save_checkpoint_with_results(
                    output_file, checkpoint, results, current_processed,
                    performance_log, cumulative_stats, verbose=True
                )

        except Exception as e:
            print(f"   ❌ Query批次 {i + 1} 处理失败: {e}")
            raise

    print(f"✅ Query向量处理完成，总API调用时间: {query_api_time_total:.2f}s, "
          f"总Token: {query_total_tokens:,}\n")

    # 处理document向量
    print("📄 正在处理document向量...")
    doc_api_time_total = 0.0
    doc_vectors = []
    doc_total_prompt_tokens = 0
    doc_total_tokens = 0

    num_batches = (len(documents) + batch_size - 1) // batch_size
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(documents))
        batch_docs = documents[start_idx:end_idx]

        try:
            batch_vectors, api_time, prompt_tokens, total_tokens, should_report = process_batch(
                batch_docs, "document", i, model, client, batch_size,
                processed_count, len(data), performance_log, cumulative_stats, report_interval
            )

            doc_vectors.extend(batch_vectors)
            doc_api_time_total += api_time
            doc_total_prompt_tokens += prompt_tokens
            doc_total_tokens += total_tokens

            # 更新结果（填充document向量）
            for j, dv in enumerate(batch_vectors):
                idx = processed_count + start_idx + j
                if idx < len(results):
                    results[idx]["document_vector"] = dv
                else:
                    # 这种情况不应该发生，但为了安全还是处理
                    print(f"   ⚠️  警告: 索引 {idx} 超出结果列表范围")

            # 在打印性能报告时保存检查点（逻辑分离：先打印报告，再保存检查点）
            if should_report:
                current_processed = processed_count + end_idx
                # 打印性能报告
                print_performance_report(
                    performance_log, current_processed, len(data), "document", model, cumulative_stats
                )
                # 保存检查点
                save_checkpoint_with_results(
                    output_file, checkpoint, results, current_processed,
                    performance_log, cumulative_stats, verbose=True
                )

        except Exception as e:
            print(f"   ❌ Document批次 {i + 1} 处理失败: {e}")
            raise

    print(f"✅ Document向量处理完成，总API调用时间: {doc_api_time_total:.2f}s, "
          f"总Token: {doc_total_tokens:,}\n")

    # 验证结果完整性
    if len(results) != len(data):
        print(f"⚠️  警告: 结果数量 ({len(results)}) 与数据数量 ({len(data)}) 不匹配")
        # 补齐缺失的结果
        for i in range(len(results), len(data)):
            results.append({
                "query": data[i].get("query", ""),
                "document": data[i].get("document", ""),
                "query_vector": None,
                "document_vector": None,
                "score": data[i].get("score")
            })

    # 计算总体性能统计
    total_api_time = query_api_time_total + doc_api_time_total
    final_total_tokens = cumulative_stats["total_tokens"]
    final_total_cost = cumulative_stats["total_cost"]

    performance_summary = {
        "total_items": len(data) * 2,  # query + document
        "query_items": len(data),
        "document_items": len(data),
        "total_api_time_seconds": round(total_api_time, 2),
        "query_api_time_seconds": round(query_api_time_total, 2),
        "document_api_time_seconds": round(doc_api_time_total, 2),
        "items_per_second": round(len(remaining_data) * 2 / total_api_time, 2) if total_api_time > 0 else 0,
        "token_usage": {
            "total_prompt_tokens": cumulative_stats["query_prompt_tokens"] + cumulative_stats["document_prompt_tokens"],
            "total_tokens": final_total_tokens,
            "query_prompt_tokens": cumulative_stats["query_prompt_tokens"],
            "query_total_tokens": cumulative_stats["query_total_tokens"],
            "document_prompt_tokens": cumulative_stats["document_prompt_tokens"],
            "document_total_tokens": cumulative_stats["document_total_tokens"],
            "avg_tokens_per_item": round(final_total_tokens / len(data) / 2, 2) if len(data) > 0 else 0
        },
        "cost": {
            "total_cost_usd": round(final_total_cost, 4),
            "query_cost_usd": round(calculate_cost(cumulative_stats["query_total_tokens"], model), 4),
            "document_cost_usd": round(calculate_cost(cumulative_stats["document_total_tokens"], model), 4),
            "price_per_million_tokens": get_model_pricing(model)
        },
        "model": model,
        "batch_size": batch_size,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # 保存最终结果
    print("💾 正在保存结果...")
    final_output = {
        "metadata": {
            "input_file": input_file,
            "model": model,
            "total_items": len(data),
            "processed_items": len(results),
            "vector_dimension": len(query_vectors[0]) if query_vectors else 0,
            "performance": performance_summary
        },
        "results": results,
        "performance_log": performance_log
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"✅ 结果已保存到: {output_file}")

    # 删除检查点文件（处理完成）
    checkpoint_file = output_file.replace('.json', '_checkpoint.json')
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"🗑️  已删除检查点文件")

    # 打印最终性能统计
    print("\n" + "=" * 60)
    print("最终性能统计（仅API调用时间）")
    print("=" * 60)
    print(f"总处理项数: {len(data)} (query: {len(data)}, document: {len(data)})")
    print(f"总API调用时间: {total_api_time:.2f}s")
    print(f"  - Query API调用: {query_api_time_total:.2f}s")
    print(f"  - Document API调用: {doc_api_time_total:.2f}s")
    print(f"处理速度: {performance_summary['items_per_second']:.2f} 项/秒")
    print(f"\nToken消耗统计:")
    print(f"  总输入Token: {performance_summary['token_usage']['total_prompt_tokens']:,}")
    print(f"  总Token: {final_total_tokens:,}")
    print(f"  - Query Token: {cumulative_stats['query_total_tokens']:,}")
    print(f"  - Document Token: {cumulative_stats['document_total_tokens']:,}")
    print(f"  平均每项Token: {performance_summary['token_usage']['avg_tokens_per_item']:.1f}")
    print(f"\n费用统计:")
    print(f"  模型: {model}")
    print(f"  定价: ${get_model_pricing(model):.4f} / 百万tokens")
    print(f"  总费用: ${final_total_cost:.4f}")
    print(f"  - Query费用: ${performance_summary['cost']['query_cost_usd']:.4f}")
    print(f"  - Document费用: ${performance_summary['cost']['document_cost_usd']:.4f}")
    print(f"\n向量维度: {len(query_vectors[0]) if query_vectors else 0}")
    print("=" * 60)


# ==================== 主函数 ====================

def main():
    """主函数，解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="为QA数据生成向量（query和document）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本使用
  python vector/vectorize/openai_test.py -i .data/mteb/nfcorpus.json -o .data/vectors/nfcorpus_vectors.json

  # 指定模型和批量大小
  python vector/vectorize/openai_test.py -i .data/mteb/nfcorpus.json -o .data/vectors/nfcorpus_vectors.json -m text-embedding-3-large -b 50

  # 从头开始处理（忽略已有检查点）
  python vector/vectorize/openai_test.py -i .data/mteb/nfcorpus.json -o .data/vectors/nfcorpus_vectors.json --restart

  # 调整性能报告间隔
  python vector/vectorize/openai_test.py -i .data/mteb/nfcorpus.json -o .data/vectors/nfcorpus_vectors.json -r 10
        """
    )

    parser.add_argument(
        '-i', '--input',
        required=True,
        help='输入JSON文件路径（包含query和document字段）'
    )

    parser.add_argument(
        '-o', '--output',
        required=True,
        help='输出JSON文件路径（包含向量和性能数据）'
    )

    parser.add_argument(
        '-m', '--model',
        default='text-embedding-3-small',
        help='使用的模型（默认: text-embedding-3-small）'
    )

    parser.add_argument(
        '-b', '--batch-size',
        type=int,
        default=100,
        help='批量处理大小（默认: 100）'
    )

    parser.add_argument(
        '--restart', '--from-scratch',
        dest='from_scratch',
        action='store_true',
        help='从头开始处理（忽略已有检查点），但运行过程中仍会保存检查点以便中断后恢复'
    )

    parser.add_argument(
        '-r', '--report-interval',
        type=int,
        default=5,
        help='性能报告打印间隔（每N个批次，默认: 5）'
    )

    args = parser.parse_args()

    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        return

    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    try:
        process_qa_data(
            input_file=args.input,
            output_file=args.output,
            model=args.model,
            batch_size=args.batch_size,
            from_scratch=args.from_scratch,
            report_interval=args.report_interval
        )
    except KeyboardInterrupt:
        print("\n⚠️  用户中断，已保存检查点，可以继续运行")
    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
