"""
暴力法基准计算脚本

使用暴力法计算query和document向量的top-N最近邻，作为向量数据库评测的基准。

使用方法:
    python vector/vector_db/brute_force_benchmark.py -i .data/vectors/nfcorpus_openai_vectors.json
    python vector/vector_db/brute_force_benchmark.py -i .data/vectors/nfcorpus_gemini_vec.json -n 20
"""
import os
import json
import time
import argparse
import hashlib
import numpy as np
from typing import List, Dict, Any, Tuple, Optional


# ==================== 工具函数 ====================

def compute_sha2048(text: str) -> str:
    """
    计算文本的SHA-2048哈希值
    注意：SHA-2048实际上是指SHA-512算法（产生512位=64字节的哈希值）
    这里使用SHA-512算法
    """
    return hashlib.sha512(text.encode('utf-8')).hexdigest()


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """计算两个向量的余弦相似度"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot_product / (norm1 * norm2))


def brute_force_top_n(
    query_vec: List[float],
    doc_vectors: List[List[float]],
    doc_ids: List[str],
    top_n: int
) -> List[Tuple[str, float]]:
    """
    使用暴力法计算top-N最近邻向量

    Args:
        query_vec: 查询向量
        doc_vectors: 文档向量列表
        doc_ids: 文档ID列表（与doc_vectors对应）
        top_n: 返回top-N结果

    Returns:
        List of (doc_id, similarity_score) tuples, sorted by similarity descending
    """
    similarities = []
    for doc_id, doc_vec in zip(doc_ids, doc_vectors):
        sim = cosine_similarity(query_vec, doc_vec)
        similarities.append((doc_id, sim))

    # 按相似度降序排序
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]


# ==================== 数据加载 ====================

def load_vector_data(vector_file: str) -> Tuple[Dict[str, Any], str]:
    """
    加载向量数据文件

    Returns:
        (数据字典（包含results和metadata）, 模型名称)
    """
    print(f"📖 正在加载向量数据: {vector_file}")
    with open(vector_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 获取模型名称
    model_name = data.get('metadata', {}).get('model', 'unknown')
    results = data.get('results', [])

    print(f"✅ 已加载 {len(results)} 条数据")
    print(f"   模型: {model_name}")

    return data, model_name


def load_original_data(original_file: str) -> List[Dict[str, Any]]:
    """加载原始QA数据文件"""
    if not os.path.exists(original_file):
        return []

    print(f"📖 正在加载原始数据: {original_file}")
    with open(original_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✅ 已加载 {len(data)} 条原始数据")
    return data


def extract_query_document_vectors(
    data: List[Dict[str, Any]],
    original_data_file: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    从数据中提取query和document向量

    Args:
        data: 向量数据列表
        original_data_file: 原始QA数据文件路径（用于匹配query和document）

    Returns:
        (query列表, document列表)
    """
    queries = []
    documents = []

    # 首先尝试直接提取query_vector和document_vector
    for item in data:
        if 'query_vector' in item and item['query_vector'] is not None:
            queries.append({
                'query': item.get('query', ''),
                'vector': item['query_vector'],
                'document': item.get('document', ''),
                'score': item.get('score')
            })

        if 'document_vector' in item and item['document_vector'] is not None:
            doc_text = item.get('document', '')
            doc_hash = compute_sha2048(doc_text)
            documents.append({
                'document': doc_text,
                'vector': item['document_vector'],
                'hash': doc_hash
            })

    # 如果数据格式不同（例如只有text_embedding），尝试从原始数据匹配
    if not queries and not documents:
        print("⚠️  未找到query_vector/document_vector字段，尝试从原始数据匹配...")

        # 加载原始数据
        original_data = []
        if original_data_file:
            if os.path.exists(original_data_file):
                original_data = load_original_data(original_data_file)
            else:
                # 尝试相对路径
                possible_paths = [
                    original_data_file,
                    os.path.join('.data', 'mteb', os.path.basename(original_data_file)),
                    os.path.join('.data/mteb', os.path.basename(original_data_file))
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        original_data = load_original_data(path)
                        break

        if not original_data:
            print("❌ 无法找到原始数据文件，无法匹配query和document")
            return queries, documents

        # 构建文本到向量的映射
        text_to_vector = {}
        for item in data:
            text = item.get('text', '')
            vector = item.get('text_embedding') or item.get('vector')
            if text and vector:
                text_to_vector[text] = vector

        # 从原始数据中匹配query和document
        query_texts_seen = set()
        doc_texts_seen = set()

        for orig_item in original_data:
            query_text = orig_item.get('query', '')
            doc_text = orig_item.get('document', '')

            # 匹配query向量
            if query_text and query_text in text_to_vector and query_text not in query_texts_seen:
                queries.append({
                    'query': query_text,
                    'vector': text_to_vector[query_text],
                    'document': doc_text,
                    'score': orig_item.get('score')
                })
                query_texts_seen.add(query_text)

            # 匹配document向量
            if doc_text and doc_text in text_to_vector and doc_text not in doc_texts_seen:
                doc_hash = compute_sha2048(doc_text)
                documents.append({
                    'document': doc_text,
                    'vector': text_to_vector[doc_text],
                    'hash': doc_hash
                })
                doc_texts_seen.add(doc_text)

    print(f"📊 提取结果:")
    print(f"   Query数量: {len(queries)}")
    print(f"   Document数量: {len(documents)}")

    # 检查数量是否合理（通常query数量应该少于document数量）
    if len(queries) > 0 and len(documents) > 0:
        if len(queries) > len(documents) * 2:
            print(f"⚠️  警告: Query数量 ({len(queries)}) 明显大于Document数量 ({len(documents)})")
            print(f"   这可能是异常的，请检查数据是否正确")
        elif len(documents) > len(queries) * 10:
            print(f"ℹ️  信息: Document数量 ({len(documents)}) 远大于Query数量 ({len(queries)})，这是正常的")
    elif len(queries) == 0:
        print(f"❌ 错误: 未找到任何query向量")
    elif len(documents) == 0:
        print(f"❌ 错误: 未找到任何document向量")
        print(f"   这可能是因为向量数据文件不完整，只包含了query向量而没有document向量")

    return queries, documents


# ==================== 基准计算 ====================

def calculate_benchmark(
    queries: List[Dict[str, Any]],
    documents: List[Dict[str, Any]],
    top_n: int
) -> Tuple[List[List[Tuple[str, float]]], List[float]]:
    """
    使用暴力法计算基准

    Args:
        queries: query列表，每个包含vector字段
        documents: document列表，每个包含hash和vector字段
        top_n: top-N检索数量

    Returns:
        (每个query的top-N结果列表, 每个query的计算时间列表)
    """
    print(f"🔍 开始计算基准（暴力法）...")
    print(f"   Query数量: {len(queries)}")
    print(f"   Document数量: {len(documents)}")
    print(f"   Top-N: {top_n}")

    # 准备数据
    doc_vectors = [doc['vector'] for doc in documents]
    doc_ids = [doc['hash'] for doc in documents]

    ground_truth = []
    computation_times = []

    total_start = time.time()

    for i, query in enumerate(queries):
        query_start = time.time()

        query_vec = query['vector']
        top_n_results = brute_force_top_n(query_vec, doc_vectors, doc_ids, top_n)
        ground_truth.append(top_n_results)

        query_time = time.time() - query_start
        computation_times.append(query_time)

        if (i + 1) % 100 == 0 or (i + 1) == len(queries):
            elapsed = time.time() - total_start
            avg_time = elapsed / (i + 1)
            remaining = avg_time * (len(queries) - i - 1)
            print(f"   进度: {i + 1}/{len(queries)} | "
                  f"平均时间: {avg_time:.4f}s | "
                  f"预计剩余: {remaining:.1f}s", end='\r')

    total_time = time.time() - total_start
    print(f"\n✅ 基准计算完成")
    print(f"   总耗时: {total_time:.2f}秒")
    print(f"   平均每个query: {total_time / len(queries):.4f}秒")

    return ground_truth, computation_times


def calculate_time_statistics(times: List[float]) -> Dict[str, float]:
    """计算时间统计信息"""
    if not times:
        return {}

    times_array = np.array(times)
    return {
        'min': float(np.min(times_array)),
        'max': float(np.max(times_array)),
        'mean': float(np.mean(times_array)),
        'median': float(np.median(times_array)),
        'p25': float(np.percentile(times_array, 25)),
        'p75': float(np.percentile(times_array, 75)),
        'p95': float(np.percentile(times_array, 95)),
        'p99': float(np.percentile(times_array, 99))
    }


# ==================== 主函数 ====================

def run_brute_force_benchmark(
    vector_file: str,
    top_n: int = 10,
    output_file: Optional[str] = None
):
    """
    运行暴力法基准计算

    Args:
        vector_file: 向量数据文件路径
        top_n: top-N检索数量
        output_file: 输出文件路径（如果为None，则自动生成）
    """
    print("=" * 80)
    print("🚀 暴力法基准计算")
    print("=" * 80)

    # 1. 加载数据
    vector_data, model_name = load_vector_data(vector_file)
    results = vector_data.get('results', [])
    metadata = vector_data.get('metadata', {})

    # 尝试获取原始数据文件路径
    original_file = metadata.get('input_file', '')
    queries, documents = extract_query_document_vectors(results, original_file)

    if not queries:
        print("❌ 未找到query向量，无法进行基准计算")
        return

    if not documents:
        print("❌ 未找到document向量，无法进行基准计算")
        return

    # 去重documents（基于hash）
    unique_docs = {}
    for doc in documents:
        doc_hash = doc['hash']
        if doc_hash not in unique_docs:
            unique_docs[doc_hash] = doc

    documents = list(unique_docs.values())
    print(f"📊 去重后Document数量: {len(documents)}")

    # 2. 计算基准
    print("\n" + "=" * 80)
    ground_truth, computation_times = calculate_benchmark(queries, documents, top_n)

    # 3. 计算统计信息
    print("\n" + "=" * 80)
    print("📊 统计信息")
    print("=" * 80)

    time_stats = calculate_time_statistics(computation_times)

    print(f"\n⏱️  计算时间统计（每个query的时间，单位：秒）:")
    if time_stats:
        print(f"   最小值: {time_stats['min']:.6f}")
        print(f"   最大值: {time_stats['max']:.6f}")
        print(f"   平均值: {time_stats['mean']:.6f}")
        print(f"   中位数: {time_stats['median']:.6f}")
        print(f"   P25: {time_stats['p25']:.6f}")
        print(f"   P75: {time_stats['p75']:.6f}")
        print(f"   P95: {time_stats['p95']:.6f}")
        print(f"   P99: {time_stats['p99']:.6f}")

    print(f"\n📈 数据统计:")
    print(f"   Query数量: {len(queries)}")
    print(f"   Document数量: {len(documents)}")
    print(f"   Top-N: {top_n}")
    print(f"   模型: {model_name}")

    # 4. 保存结果
    if output_file is None:
        output_file = vector_file.replace('.json', '_brute_force_benchmark.json')

    # 格式化结果（只保存doc_id，不保存相似度分数，以节省空间）
    ground_truth_ids = [[doc_id for doc_id, _ in results] for results in ground_truth]

    results = {
        'vector_file': vector_file,
        'model': model_name,
        'query_count': len(queries),
        'document_count': len(documents),
        'top_n': top_n,
        'ground_truth': ground_truth_ids,  # 只保存ID列表
        'ground_truth_with_scores': ground_truth,  # 完整结果（包含分数）
        'time_statistics': time_stats,
        'computation_times': [float(t) for t in computation_times],
        'metadata': {
            'queries': [
                {
                    'query': q['query'],
                    'document': q.get('document', ''),
                    'score': q.get('score')
                }
                for q in queries
            ]
        }
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 结果已保存到: {output_file}")
    print("=" * 80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="暴力法基准计算脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本使用
  python vector/vector_db/brute_force_benchmark.py -i .data/vectors/nfcorpus_openai_vectors.json

  # 指定top-N
  python vector/vector_db/brute_force_benchmark.py -i .data/vectors/nfcorpus_gemini_vec.json -n 20

  # 指定输出文件
  python vector/vector_db/brute_force_benchmark.py -i .data/vectors/nfcorpus_gemini_multimodal_vec.json -o benchmark_results.json
        """
    )

    parser.add_argument(
        '-i', '--input',
        required=True,
        help='向量数据文件路径（JSON格式）'
    )

    parser.add_argument(
        '-n', '--top-n',
        type=int,
        default=10,
        help='Top-N检索数量（默认: 10）'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='输出文件路径（默认: 输入文件名_brute_force_benchmark.json）'
    )

    args = parser.parse_args()

    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        return

    try:
        run_brute_force_benchmark(
            vector_file=args.input,
            top_n=args.top_n,
            output_file=args.output
        )
    except KeyboardInterrupt:
        print("\n⚠️  用户中断")
    except Exception as e:
        print(f"\n❌ 计算失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

