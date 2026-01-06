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

def load_qa_data(input_file: str) -> Dict[str, Any]:
    """
    加载QA数据
    支持两种格式：
    1. 新格式: {"query_list": [...], "document_list": [...]}
    2. 旧格式: [{"query": "...", "document": "...", "score": ...}]
    """
    print(f"📖 正在加载数据: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 检测数据格式
    if isinstance(data, dict) and "query_list" in data and "document_list" in data:
        # 新格式：{"query_list": [...], "document_list": [...]}
        query_list = data.get("query_list", [])
        document_list = data.get("document_list", [])
        print(f"✅ 已加载新格式数据: {len(query_list)} 条唯一query, {len(document_list)} 条唯一document")
        return {
            "format": "new",
            "query_list": query_list,
            "document_list": document_list
        }
    elif isinstance(data, list):
        # 旧格式：列表格式
        print(f"✅ 已加载旧格式数据: {len(data)} 条数据")
        return {
            "format": "old",
            "data": data
        }
    else:
        raise ValueError(f"不支持的数据格式。期望格式: {{\"query_list\": [...], \"document_list\": [...]}} 或 [{{\"query\": ..., \"document\": ...}}]")


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


def get_default_checkpoint() -> Dict[str, Any]:
    """获取默认的检查点结构"""
    return {
        "query_processed_count": 0,
        "document_processed_count": 0,
        "results": [],
        "performance": [],
        "cumulative_stats": get_default_cumulative_stats(),
        "query_vector_cache": {},  # 用于存储去重后的query向量
        "document_vector_cache": {}  # 用于存储去重后的document向量
    }


def load_checkpoint(output_file: str) -> Dict[str, Any]:
    """加载检查点（已处理的数据，包含性能和token统计）"""
    checkpoint_file = output_file.replace('.json', '_checkpoint.json')
    if os.path.exists(checkpoint_file):
        print(f"📂 发现检查点文件: {checkpoint_file}")
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)

            # 兼容旧格式（只有processed_count）
            if 'processed_count' in checkpoint and 'query_processed_count' not in checkpoint:
                # 旧格式，转换为新格式
                old_count = checkpoint.get('processed_count', 0)
                checkpoint['query_processed_count'] = old_count
                checkpoint['document_processed_count'] = old_count
                del checkpoint['processed_count']

            query_processed = checkpoint.get('query_processed_count', 0)
            document_processed = checkpoint.get('document_processed_count', 0)
            print(f"   Query已处理: {query_processed} 条")
            print(f"   Document已处理: {document_processed} 条")

            # 恢复累计的token统计
            cumulative_stats = checkpoint.get('cumulative_stats', get_default_cumulative_stats())
            if cumulative_stats.get("total_tokens", 0) > 0:
                print(f"   累计Token: {cumulative_stats['total_tokens']:,}")
                print(f"   累计费用: ${cumulative_stats.get('total_cost', 0):.4f}")

            # 恢复query向量缓存（用于去重）
            if 'query_vector_cache' not in checkpoint:
                checkpoint['query_vector_cache'] = {}
            # 恢复document向量缓存（用于去重）
            if 'document_vector_cache' not in checkpoint:
                checkpoint['document_vector_cache'] = {}

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
            return get_default_checkpoint()
        except Exception as e:
            print(f"   ❌ 加载检查点文件时出错: {e}")
            print(f"   💡 将从头开始处理")
            return get_default_checkpoint()

    return get_default_checkpoint()


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
    processed_unique_count: int,
    unique_total_count: int,
    performance_log: List[Dict[str, Any]],
    cumulative_stats: Dict[str, Any],
    report_interval: int
) -> Tuple[List[List[float]], float, int, int, bool]:
    """
    处理一批文本，返回向量、API时间、token统计

    Args:
        processed_unique_count: 已处理的唯一项数量（用于进度显示分子）
        unique_total_count: 去重后的总数（用于进度显示分母）

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

    # 打印进度：使用实际已处理的唯一项数量作为分子，去重后的总数作为分母
    processed = processed_unique_count + len(texts)
    if processed > unique_total_count:
        processed = unique_total_count
    progress = processed / unique_total_count * 100 if unique_total_count > 0 else 0
    print(f"   {text_type.capitalize()}批次 {batch_index + 1}: {len(texts)} 条, "
          f"API调用时间 {api_time:.2f}s, "
          f"Token: {token_info.get('total_tokens', 0):,}, "
          f"总进度: {processed}/{unique_total_count} ({progress:.1f}%)")

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
    query_processed_count: int,
    document_processed_count: int,
    performance_log: List[Dict[str, Any]],
    cumulative_stats: Dict[str, Any],
    query_vector_cache: Optional[Dict[str, List[float]]] = None,
    document_vector_cache: Optional[Dict[str, List[float]]] = None,
    verbose: bool = False
):
    """
    保存检查点（包含结果）
    注意：无论是否从头开始，运行过程中都会保存检查点，以便中断后可以恢复
    """
    checkpoint["results"] = results
    checkpoint["query_processed_count"] = query_processed_count
    checkpoint["document_processed_count"] = document_processed_count
    checkpoint["performance"] = performance_log
    checkpoint["cumulative_stats"] = cumulative_stats
    if query_vector_cache is not None:
        checkpoint["query_vector_cache"] = query_vector_cache
    if document_vector_cache is not None:
        checkpoint["document_vector_cache"] = document_vector_cache
    save_checkpoint(output_file, checkpoint, verbose=verbose)


def process_qa_data(
    input_file: str,
    output_file: str,
    model: str = "text-embedding-3-small",
    batch_size: int = 100,
    from_scratch: bool = False,
    report_interval: int = 5,
    max_items: Optional[int] = None
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
        max_items: 最大处理条数（None表示处理所有数据）
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
    if max_items:
        print(f"最大处理条数: {max_items}")
    print("=" * 60)

    # 加载数据
    loaded_data = load_qa_data(input_file)

    # 根据数据格式处理
    if loaded_data["format"] == "new":
        # 新格式：已经去重的列表
        query_list = loaded_data["query_list"]
        document_list = loaded_data["document_list"]

        # 如果指定了最大条数，则截取数据
        if max_items and max_items > 0:
            original_query_count = len(query_list)
            original_doc_count = len(document_list)
            query_list = query_list[:max_items]
            document_list = document_list[:max_items]
            print(f"📊 数据限制: Query从 {original_query_count} 条限制到 {len(query_list)} 条")
            print(f"📊 数据限制: Document从 {original_doc_count} 条限制到 {len(document_list)} 条")

        # 转换为旧格式的兼容结构（用于后续处理）
        data_format = "new"
        data = None  # 新格式不需要 data 列表
    else:
        # 旧格式：列表格式
        data = loaded_data["data"]
        data_format = "old"

        # 如果指定了最大条数，则截取数据
        original_data_count = len(data)
        if max_items and max_items > 0:
            data = data[:max_items]
            print(f"📊 数据限制: 从 {original_data_count} 条限制到 {len(data)} 条")

        # 从旧格式中提取 query_list 和 document_list
        query_list = [item["query"] for item in data if item.get("query")]
        document_list = [item["document"] for item in data if item.get("document")]

    # 加载检查点（如果 from_scratch=True，则忽略已有检查点，从头开始）
    if from_scratch:
        checkpoint = get_default_checkpoint()
        # 如果存在旧的检查点文件，提示用户
        checkpoint_file = output_file.replace('.json', '_checkpoint.json')
        if os.path.exists(checkpoint_file):
            print(f"⚠️  发现已有检查点文件，但将从头开始处理（检查点文件不会被删除）")
    else:
        checkpoint = load_checkpoint(output_file)

    query_processed_count = checkpoint.get("query_processed_count", 0)
    document_processed_count = checkpoint.get("document_processed_count", 0)
    results = checkpoint.get("results", [])
    performance_log = checkpoint.get("performance", [])
    cumulative_stats = checkpoint.get("cumulative_stats", get_default_cumulative_stats())
    query_vector_cache = checkpoint.get("query_vector_cache", {})
    document_vector_cache = checkpoint.get("document_vector_cache", {})

    if query_processed_count > 0 or document_processed_count > 0:
        print(f"🔄 断点续跑状态:")
        print(f"   Query: {query_processed_count}/{len(query_list)} 条已处理")
        print(f"   Document: {document_processed_count}/{len(document_list)} 条已处理")
        if cumulative_stats.get("total_tokens", 0) > 0:
            print(f"   累计Token: {cumulative_stats['total_tokens']:,}")
            print(f"   累计费用: ${cumulative_stats.get('total_cost', 0):.4f}")

    # 获取客户端
    client = get_openai_client()

    # 处理剩余的数据（新格式已经是去重后的列表）
    query_remaining_data = query_list[query_processed_count:]
    document_remaining_data = document_list[document_processed_count:]

    if len(query_remaining_data) == 0 and len(document_remaining_data) == 0:
        print("✅ 所有数据已处理完成！")
        return

    print(f"\n📊 处理进度:")
    print(f"   Query: {query_processed_count}/{len(query_list)} ({query_processed_count/len(query_list)*100:.1f}%), 剩余: {len(query_remaining_data)} 条")
    print(f"   Document: {document_processed_count}/{len(document_list)} ({document_processed_count/len(document_list)*100:.1f}%), 剩余: {len(document_remaining_data)} 条\n")

    # 新格式已经是去重后的列表，直接使用
    unique_query_list = query_remaining_data
    unique_document_list = document_remaining_data

    print(f"🔍 Query去重分析: 数据已去重，共 {len(unique_query_list)} 条唯一query")
    print(f"🔍 Document去重分析: 数据已去重，共 {len(unique_document_list)} 条唯一document")

    # 处理query向量（使用去重后的唯一query）
    print("🔍 正在处理query向量（已去重）...")
    query_api_time_total = 0.0
    unique_query_vectors = {}  # 存储唯一query的向量
    query_total_prompt_tokens = 0
    query_total_tokens = 0

    # 检查缓存中已有的query向量
    cached_count = 0
    for query in unique_query_list:
        if query in query_vector_cache:
            unique_query_vectors[query] = query_vector_cache[query]
            cached_count += 1

    if cached_count > 0:
        print(f"   💾 从缓存中恢复 {cached_count} 个query向量")

    # 只处理未缓存的唯一query
    uncached_unique_queries = [q for q in unique_query_list if q not in unique_query_vectors]
    last_batch_was_reported = False

    if uncached_unique_queries:
        num_batches = (len(uncached_unique_queries) + batch_size - 1) // batch_size
        processed_unique_queries = 0  # 跟踪已处理的唯一query数量

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(uncached_unique_queries))
            batch_queries = uncached_unique_queries[start_idx:end_idx]

            try:
                batch_vectors, api_time, prompt_tokens, total_tokens, should_report = process_batch(
                    batch_queries, "query", i, model, client, batch_size,
                    processed_unique_queries, len(uncached_unique_queries), performance_log, cumulative_stats, report_interval
                )

                # 更新已处理的唯一query数量
                processed_unique_queries += len(batch_queries)

                query_api_time_total += api_time
                query_total_prompt_tokens += prompt_tokens
                query_total_tokens += total_tokens

                # 存储唯一query的向量到缓存
                for j, qv in enumerate(batch_vectors):
                    query_text = batch_queries[j]
                    unique_query_vectors[query_text] = qv
                    query_vector_cache[query_text] = qv

                # 在打印性能报告时保存检查点（逻辑分离：先打印报告，再保存检查点）
                if should_report:
                    # 计算已处理的query数量
                    current_query_processed = query_processed_count + processed_unique_queries
                    # 打印性能报告
                    print_performance_report(
                        performance_log, current_query_processed, len(query_list), "query", model, cumulative_stats
                    )
                    # 保存检查点（包含query向量缓存）
                    save_checkpoint_with_results(
                        output_file, checkpoint, results,
                        current_query_processed,
                        document_processed_count,
                        performance_log, cumulative_stats, query_vector_cache, document_vector_cache, verbose=True
                    )
                    last_batch_was_reported = True
                elif i == num_batches - 1:
                    # 最后一批，即使不满足报告间隔也要报告
                    last_batch_was_reported = False

            except Exception as e:
                print(f"   ❌ Query批次 {i + 1} 处理失败: {e}")
                raise
    else:
        print(f"   ✅ 所有query向量已从缓存中恢复，无需API调用")

    # Query处理完成，如果最后一批没有报告，则输出最终报告
    final_query_processed = query_processed_count + len(query_remaining_data)
    if not last_batch_was_reported and (len(uncached_unique_queries) > 0 or cached_count > 0):
        # 打印最终query性能报告
        print_performance_report(
            performance_log, final_query_processed, len(query_list), "query", model, cumulative_stats
        )
        # 保存检查点
        save_checkpoint_with_results(
            output_file, checkpoint, results,
            final_query_processed,
            document_processed_count,
            performance_log, cumulative_stats, query_vector_cache, document_vector_cache, verbose=True
        )

    print(f"✅ Query向量处理完成，总API调用时间: {query_api_time_total:.2f}s, "
          f"总Token: {query_total_tokens:,}, "
          f"实际调用: {len(uncached_unique_queries)} 条唯一query\n")

    # 处理document向量
    print("📄 正在处理document向量...")

    # 新格式已经是去重后的列表，直接使用
    unique_document_list = document_remaining_data

    doc_api_time_total = 0.0
    unique_document_vectors = {}  # 存储唯一document的向量
    doc_total_prompt_tokens = 0
    doc_total_tokens = 0

    # 检查缓存中已有的document向量
    cached_doc_count = 0
    for document in unique_document_list:
        if document in document_vector_cache:
            unique_document_vectors[document] = document_vector_cache[document]
            cached_doc_count += 1

    if cached_doc_count > 0:
        print(f"   💾 从缓存中恢复 {cached_doc_count} 个document向量")

    # 只处理未缓存的唯一document
    uncached_unique_documents = [d for d in unique_document_list if d not in unique_document_vectors]

    num_batches = (len(uncached_unique_documents) + batch_size - 1) // batch_size if uncached_unique_documents else 0
    last_doc_batch_was_reported = False

    if uncached_unique_documents:
        processed_unique_documents = 0  # 跟踪已处理的唯一document数量

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(uncached_unique_documents))
            batch_docs = uncached_unique_documents[start_idx:end_idx]

            try:
                batch_vectors, api_time, prompt_tokens, total_tokens, should_report = process_batch(
                    batch_docs, "document", i, model, client, batch_size,
                    processed_unique_documents, len(uncached_unique_documents), performance_log, cumulative_stats, report_interval
                )

                # 更新已处理的唯一document数量
                processed_unique_documents += len(batch_docs)

                doc_api_time_total += api_time
                doc_total_prompt_tokens += prompt_tokens
                doc_total_tokens += total_tokens

                # 存储唯一document的向量到缓存
                for j, dv in enumerate(batch_vectors):
                    document_text = batch_docs[j]
                    unique_document_vectors[document_text] = dv
                    document_vector_cache[document_text] = dv

                # 在打印性能报告时保存检查点（逻辑分离：先打印报告，再保存检查点）
                if should_report:
                    current_doc_processed = document_processed_count + processed_unique_documents
                    # 打印性能报告
                    print_performance_report(
                        performance_log, current_doc_processed, len(document_list), "document", model, cumulative_stats
                    )
                    # 保存检查点
                    save_checkpoint_with_results(
                        output_file, checkpoint, results,
                        final_query_processed,  # 所有query都已处理
                        current_doc_processed,
                        performance_log, cumulative_stats, query_vector_cache, document_vector_cache, verbose=True
                    )
                    last_doc_batch_was_reported = True
                elif i == num_batches - 1:
                    # 最后一批，即使不满足报告间隔也要报告
                    last_doc_batch_was_reported = False

            except Exception as e:
                print(f"   ❌ Document批次 {i + 1} 处理失败: {e}")
                raise
    else:
        print(f"   ✅ 所有document向量已从缓存中恢复，无需API调用")

    # Document处理完成，如果最后一批没有报告，则输出最终报告
    final_doc_processed = document_processed_count + len(document_remaining_data)
    if not last_doc_batch_was_reported and (num_batches > 0 or cached_doc_count > 0):
        # 打印最终document性能报告
        print_performance_report(
            performance_log, final_doc_processed, len(document_list), "document", model, cumulative_stats
        )
        # 保存检查点
        save_checkpoint_with_results(
            output_file, checkpoint, results,
            final_query_processed,
            final_doc_processed,
            performance_log, cumulative_stats, query_vector_cache, document_vector_cache, verbose=True
        )

    print(f"✅ Document向量处理完成，总API调用时间: {doc_api_time_total:.2f}s, "
          f"总Token: {doc_total_tokens:,}, "
          f"实际调用: {len(uncached_unique_documents)} 条唯一document\n")

    # 构建输出向量列表（按原始顺序）
    query_vectors = []
    document_vectors = []

    if data_format == "new":
        # 新格式：直接按顺序构建向量列表
        for query in query_list:
            query_vectors.append(query_vector_cache.get(query))
        for document in document_list:
            document_vectors.append(document_vector_cache.get(document))
    else:
        # 旧格式：从 results 中提取
        for item in data:
            query = item.get("query", "")
            document = item.get("document", "")
            query_vectors.append(query_vector_cache.get(query))
            document_vectors.append(document_vector_cache.get(document))

    # 计算总体性能统计
    total_api_time = query_api_time_total + doc_api_time_total
    final_total_tokens = cumulative_stats["total_tokens"]
    final_total_cost = cumulative_stats["total_cost"]

    total_query_count = len(query_list)
    total_doc_count = len(document_list)

    performance_summary = {
        "total_items": total_query_count + total_doc_count,
        "query_items": total_query_count,
        "document_items": total_doc_count,
        "total_api_time_seconds": round(total_api_time, 2),
        "query_api_time_seconds": round(query_api_time_total, 2),
        "document_api_time_seconds": round(doc_api_time_total, 2),
        "items_per_second": round((total_query_count + total_doc_count) / total_api_time, 2) if total_api_time > 0 else 0,
        "token_usage": {
            "total_prompt_tokens": cumulative_stats["query_prompt_tokens"] + cumulative_stats["document_prompt_tokens"],
            "total_tokens": final_total_tokens,
            "query_prompt_tokens": cumulative_stats["query_prompt_tokens"],
            "query_total_tokens": cumulative_stats["query_total_tokens"],
            "document_prompt_tokens": cumulative_stats["document_prompt_tokens"],
            "document_total_tokens": cumulative_stats["document_total_tokens"],
            "avg_tokens_per_item": round(final_total_tokens / (total_query_count + total_doc_count), 2) if (total_query_count + total_doc_count) > 0 else 0
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
    vector_dimension = len(query_vectors[0]) if query_vectors and query_vectors[0] else (len(document_vectors[0]) if document_vectors and document_vectors[0] else 0)

    final_output = {
        "metadata": {
            "input_file": input_file,
            "model": model,
            "query_count": total_query_count,
            "document_count": total_doc_count,
            "vector_dimension": vector_dimension,
            "performance": performance_summary
        },
        "query_vectors": query_vectors,
        "document_vectors": document_vectors,
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
    print(f"总处理项数: {total_query_count + total_doc_count} (query: {total_query_count}, document: {total_doc_count})")
    print(f"总API调用时间: {total_api_time:.2f}s")
    print(f"  - Query API调用: {query_api_time_total:.2f}s")
    print(f"  - Document API调用: {doc_api_time_total:.2f}s")
    print(f"处理速度: {performance_summary['items_per_second']:.2f} 项/秒")
    if total_query_count + total_doc_count > 0:
        avg_time_per_item = total_api_time / (total_query_count + total_doc_count)
        avg_query_time = query_api_time_total / total_query_count if total_query_count > 0 else 0
        avg_doc_time = doc_api_time_total / total_doc_count if total_doc_count > 0 else 0
        print(f"\n平均每条数据耗时:")
        print(f"  总耗时: {avg_time_per_item:.4f}s/条")
        print(f"  - Query耗时: {avg_query_time:.4f}s/条")
        print(f"  - Document耗时: {avg_doc_time:.4f}s/条")
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
    print(f"\n向量维度: {vector_dimension}")
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

  # 限制处理条数（用于测试）
  python vector/vectorize/openai_test.py -i .data/mteb/nfcorpus.json -o .data/vectors/nfcorpus_vectors.json --max-items 100
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

    parser.add_argument(
        '--max-items',
        type=int,
        default=None,
        help='最大处理条数（默认: 处理所有数据，用于测试或小规模运行）'
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
            report_interval=args.report_interval,
            max_items=args.max_items
        )
    except KeyboardInterrupt:
        print("\n⚠️  用户中断，已保存检查点，可以继续运行")
    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
