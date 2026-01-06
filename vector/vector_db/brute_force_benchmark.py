"""
暴力法基准计算脚本

使用暴力法计算query和document向量的top-N最近邻，作为向量数据库评测的基准。

使用方法:
    python vector/vector_db/brute_force_benchmark.py -i .data/vectors/scidocs_gemini_vectors.json
    python vector/vector_db/brute_force_benchmark.py -i .data/vectors/scidocs_gemini_vectors.json -n 20
"""
import os
import json
import time
import argparse
import hashlib
import numpy as np
from typing import List, Dict, Any, Tuple, Optional


# ==================== 工具函数 ====================

def compute_sha512_hex(text: str) -> str:
    """计算文本的SHA-512哈希值（十六进制字符串）"""
    return hashlib.sha512(text.encode('utf-8')).hexdigest()


def find_original_data_file(original_file: str) -> Optional[str]:
    """查找原始数据文件（尝试多个可能的路径）"""
    if not original_file:
        return None

    possible_paths = [
        original_file,
        os.path.join('.data', 'mteb', os.path.basename(original_file)),
        os.path.join('.data/mteb', os.path.basename(original_file))
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    return None


def brute_force_top_n(
    query_vec: np.ndarray,
    doc_vectors_normalized: np.ndarray,
    doc_ids: List[str],
    top_n: int
) -> List[Tuple[str, float]]:
    """
    使用暴力法计算top-N最近邻向量（使用NumPy加速）

    Args:
        query_vec: 查询向量（NumPy数组，已归一化）
        doc_vectors_normalized: 归一化的文档向量矩阵 [n_docs, dim]
        doc_ids: 文档ID列表
        top_n: 返回top-N结果

    Returns:
        List of (doc_id, similarity_score) tuples, sorted by similarity descending
    """
    # 归一化查询向量
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return [(doc_ids[i], 0.0) for i in range(min(top_n, len(doc_ids)))]
    query_normalized = query_vec / query_norm

    # 计算所有文档与查询向量的余弦相似度（矩阵乘法）
    similarities = np.dot(doc_vectors_normalized, query_normalized)

    # 获取top-N索引
    if top_n >= len(doc_ids):
        top_indices = np.argsort(similarities)[::-1]
    else:
        # 使用argpartition只部分排序，更快
        top_indices = np.argpartition(similarities, -top_n)[-top_n:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

    return [(doc_ids[idx], float(similarities[idx])) for idx in top_indices]


# ==================== 数据加载 ====================

def load_vector_data(vector_file: str) -> Tuple[Dict[str, Any], str]:
    """加载向量数据文件"""
    print(f"📖 正在加载向量数据: {vector_file}")
    with open(vector_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    model_name = data.get('metadata', {}).get('model', 'unknown')

    if 'query_vectors' in data and 'document_vectors' in data:
        query_count = len(data.get('query_vectors', []))
        doc_count = len(data.get('document_vectors', []))
        print(f"✅ 已加载数据（新格式）")
        print(f"   Query向量: {query_count}, Document向量: {doc_count}, 模型: {model_name}")
    elif 'results' in data:
        results = data.get('results', [])
        print(f"✅ 已加载数据（旧格式）")
        print(f"   数据条数: {len(results)}, 模型: {model_name}")
    else:
        print(f"⚠️  未知的数据格式")

    return data, model_name


def load_original_data(original_file: str):
    """加载原始QA数据文件"""
    file_path = find_original_data_file(original_file)
    if not file_path:
        return None

    print(f"📖 正在加载原始数据: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict):
        print(f"✅ 已加载原始数据（新格式）")
    elif isinstance(data, list):
        print(f"✅ 已加载原始数据（旧格式，{len(data)} 条）")
    else:
        print(f"⚠️  原始数据格式未知")

    return data


# ==================== 数据提取 ====================

def extract_unique_texts_from_original_data(original_data) -> Tuple[List[str], List[str]]:
    """从原始数据中提取唯一的query和document列表（按首次出现顺序）"""
    if isinstance(original_data, dict):
        # 新格式：直接获取列表
        if 'query_list' in original_data and 'document_list' in original_data:
            return original_data['query_list'], original_data['document_list']
        else:
            raise ValueError("新格式原始数据应包含query_list和document_list")

    elif isinstance(original_data, list):
        # 旧格式：从列表中提取唯一值
        query_list = []
        document_list = []
        query_seen = set()
        doc_seen = set()

        for item in original_data:
            query_text = item.get('query', '')
            doc_text = item.get('document', '')

            if query_text and query_text not in query_seen:
                query_list.append(query_text)
                query_seen.add(query_text)

            if doc_text:
                doc_hash = compute_sha512_hex(doc_text)
                if doc_hash not in doc_seen:
                    document_list.append(doc_text)
                    doc_seen.add(doc_hash)

        return query_list, document_list

    else:
        raise ValueError(f"未知的原始数据格式: {type(original_data)}")


def extract_vectors_new_format(
    vector_data: Dict[str, Any],
    original_data_file: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    从新格式向量数据中提取query和document向量

    注意：不能假设向量列表的索引与文本列表的索引对应。
    向量化过程中可能有失败、跳过等情况，导致顺序不一致。
    如果向量数据中包含文本信息，优先使用；否则需要从原始数据匹配。
    """
    query_vectors = vector_data.get('query_vectors', [])
    document_vectors = vector_data.get('document_vectors', [])

    # 检查向量数据中是否包含文本信息（理想情况）
    query_texts = vector_data.get('query_texts', None)
    document_texts = vector_data.get('document_texts', None)

    if query_texts is not None and document_texts is not None:
        # 向量数据中包含文本信息，直接使用
        print(f"✅ 向量数据中包含文本信息，直接匹配")
        queries = []
        for query_text, query_vec in zip(query_texts, query_vectors):
            if query_vec is not None:  # 跳过None值（向量化失败的）
                queries.append({
                    'query': query_text,
                    'vector': query_vec
                })

        documents = []
        for doc_text, doc_vec in zip(document_texts, document_vectors):
            if doc_vec is not None:  # 跳过None值
                documents.append({
                    'document': doc_text,
                    'vector': doc_vec,
                    'hash': compute_sha512_hex(doc_text)
                })

        return queries, documents

    # 向量数据中不包含文本信息，需要从原始数据匹配
    print(f"⚠️  向量数据中不包含文本信息，从原始数据匹配（可能不准确）")

    original_data = load_original_data(original_data_file)
    if not original_data:
        raise ValueError(f"无法加载原始数据文件: {original_data_file}")

    # 提取唯一的query和document文本
    query_list, document_list = extract_unique_texts_from_original_data(original_data)

    # 验证数量
    if len(query_vectors) != len(query_list):
        print(f"⚠️  警告: Query向量数量 ({len(query_vectors)}) != 唯一Query数量 ({len(query_list)})")
        print(f"   注意：向量化过程中可能有失败或跳过，索引匹配可能不准确")
    if len(document_vectors) != len(document_list):
        print(f"⚠️  警告: Document向量数量 ({len(document_vectors)}) != 唯一Document数量 ({len(document_list)})")
        print(f"   注意：向量化过程中可能有失败或跳过，索引匹配可能不准确")

    # 尝试按索引匹配（但这是不安全的假设）
    # 只匹配有效向量（非None）和对应索引的文本
    queries = []
    for i, query_text in enumerate(query_list):
        if i < len(query_vectors) and query_vectors[i] is not None:
            queries.append({
                'query': query_text,
                'vector': query_vectors[i]
            })

    documents = []
    for i, doc_text in enumerate(document_list):
        if i < len(document_vectors) and document_vectors[i] is not None:
            documents.append({
                'document': doc_text,
                'vector': document_vectors[i],
                'hash': compute_sha512_hex(doc_text)
            })

    # 如果匹配结果数量不一致，给出警告
    matched_query_count = len(queries)
    matched_doc_count = len(documents)
    valid_query_vectors = sum(1 for v in query_vectors if v is not None)
    valid_doc_vectors = sum(1 for v in document_vectors if v is not None)

    if matched_query_count != valid_query_vectors:
        print(f"⚠️  警告: 匹配的Query数量 ({matched_query_count}) != 有效Query向量数量 ({valid_query_vectors})")
        print(f"   建议：向量数据文件应包含query_texts和document_texts字段以确保准确匹配")

    if matched_doc_count != valid_doc_vectors:
        print(f"⚠️  警告: 匹配的Document数量 ({matched_doc_count}) != 有效Document向量数量 ({valid_doc_vectors})")
        print(f"   建议：向量数据文件应包含query_texts和document_texts字段以确保准确匹配")

    return queries, documents


def extract_vectors_old_format(
    results: List[Dict[str, Any]],
    original_data_file: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """从旧格式向量数据中提取query和document向量"""
    queries = []
    documents = []

    # 直接提取query_vector和document_vector
    for item in results:
        if 'query_vector' in item and item['query_vector'] is not None:
            queries.append({
                'query': item.get('query', ''),
                'vector': item['query_vector']
            })

        if 'document_vector' in item and item['document_vector'] is not None:
            doc_text = item.get('document', '')
            documents.append({
                'document': doc_text,
                'vector': item['document_vector'],
                'hash': compute_sha512_hex(doc_text)
            })

    # 如果没有找到，尝试从原始数据匹配（处理text_embedding格式）
    if not queries and not documents and original_data_file:
        print("⚠️  未找到query_vector/document_vector，尝试从原始数据匹配...")
        original_data = load_original_data(original_data_file)
        if original_data:
            # 构建文本到向量的映射
            text_to_vector = {}
            for item in results:
                text = item.get('text', '')
                vector = item.get('text_embedding') or item.get('vector')
                if text and vector:
                    text_to_vector[text] = vector

            # 从原始数据匹配
            query_list, document_list = extract_unique_texts_from_original_data(original_data)

            for query_text in query_list:
                if query_text in text_to_vector:
                    queries.append({
                        'query': query_text,
                        'vector': text_to_vector[query_text]
                    })

            for doc_text in document_list:
                if doc_text in text_to_vector:
                    documents.append({
                        'document': doc_text,
                        'vector': text_to_vector[doc_text],
                        'hash': compute_sha512_hex(doc_text)
                    })

    return queries, documents


def extract_vectors(
    vector_data: Dict[str, Any],
    original_data_file: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """提取query和document向量（自动识别格式）"""
    if 'query_vectors' in vector_data and 'document_vectors' in vector_data:
        # 新格式
        if not original_data_file:
            raise ValueError("新格式需要原始数据文件路径")
        queries, documents = extract_vectors_new_format(vector_data, original_data_file)
    else:
        # 旧格式
        results = vector_data.get('results', [])
        queries, documents = extract_vectors_old_format(results, original_data_file)

    # 验证和去重
    print(f"📊 提取结果: Query={len(queries)}, Document={len(documents)}")

    # 去重documents（基于hash）
    unique_docs = {}
    for doc in documents:
        doc_hash = doc['hash']
        if doc_hash not in unique_docs:
            unique_docs[doc_hash] = doc
    documents = list(unique_docs.values())

    # 检查数量合理性
    if len(queries) == 0:
        raise ValueError("未找到任何query向量")
    if len(documents) == 0:
        raise ValueError("未找到任何document向量")
    if len(queries) > len(documents) * 2:
        print(f"⚠️  警告: Query数量 ({len(queries)}) 明显大于Document数量 ({len(documents)})")
    elif len(documents) > len(queries) * 10:
        print(f"ℹ️  信息: Document数量 ({len(documents)}) 远大于Query数量 ({len(queries)})，这是正常的")

    print(f"📊 去重后: Query={len(queries)}, Document={len(documents)}")
    return queries, documents


# ==================== 基准计算 ====================

def calculate_benchmark(
    queries: List[Dict[str, Any]],
    documents: List[Dict[str, Any]],
    top_n: int
) -> Tuple[List[List[Tuple[str, float]]], List[float]]:
    """
    使用暴力法计算基准（NumPy加速）

    注意：这是「NumPy BLAS brute-force」，不是理论上的单核brute-force。
    实际使用的是：
    - NumPy + MKL/OpenBLAS
    - SIMD指令集优化
    - 多线程并行计算

    因此，如果后续对比 FAISS Flat 或 Milvus brute-force 的结果，
    需要谨慎考虑这些实现差异对性能的影响。
    """
    print(f"\n🔍 开始计算基准（暴力法，NumPy加速）...")
    print(f"   Query数量: {len(queries)}, Document数量: {len(documents)}, Top-N: {top_n}")

    # 准备数据：转换为NumPy矩阵并归一化
    print(f"   正在准备数据...")

    # 检查向量维度一致性（在转换为NumPy数组之前）
    if not documents:
        raise ValueError("documents列表为空")

    doc_dim = len(documents[0]['vector'])
    for i, doc in enumerate(documents):
        if len(doc['vector']) != doc_dim:
            raise ValueError(f"Document {i} (hash: {doc.get('hash', 'unknown')}) 向量维度不匹配: "
                           f"期望 {doc_dim}, 实际 {len(doc['vector'])}")

    # 检查query向量维度
    if not queries:
        raise ValueError("queries列表为空")

    query_dim = len(queries[0]['vector'])
    for i, query in enumerate(queries):
        if len(query['vector']) != query_dim:
            raise ValueError(f"Query {i} (text: {query.get('query', 'unknown')[:50]}...) 向量维度不匹配: "
                           f"期望 {query_dim}, 实际 {len(query['vector'])}")

    # 检查query和document维度是否一致
    if query_dim != doc_dim:
        raise ValueError(f"Query和Document向量维度不一致: Query={query_dim}, Document={doc_dim}")

    doc_vectors_matrix = np.array([doc['vector'] for doc in documents], dtype=np.float32)
    doc_ids = [doc['hash'] for doc in documents]

    # 归一化所有文档向量（一次性归一化）
    doc_norms = np.linalg.norm(doc_vectors_matrix, axis=1, keepdims=True)
    doc_norms[doc_norms == 0] = 1.0  # 避免除零
    doc_vectors_normalized = doc_vectors_matrix / doc_norms

    print(f"   ✅ 数据准备完成（矩阵形状: {doc_vectors_matrix.shape}）")

    # 计算基准
    ground_truth = []
    computation_times = []
    total_start = time.time()

    for i, query in enumerate(queries):
        query_start = time.time()

        query_vec = np.array(query['vector'], dtype=np.float32)
        top_n_results = brute_force_top_n(
            query_vec,
            doc_vectors_normalized,
            doc_ids,
            top_n
        )
        ground_truth.append(top_n_results)
        computation_times.append(time.time() - query_start)

        # 进度报告
        if (i + 1) % 100 == 0 or (i + 1) == len(queries):
            elapsed = time.time() - total_start
            avg_time = elapsed / (i + 1)
            remaining = avg_time * (len(queries) - i - 1)
            print(f"   进度: {i + 1}/{len(queries)} | "
                  f"平均: {avg_time:.4f}s | "
                  f"剩余: {remaining:.1f}s", end='\r')

    total_time = time.time() - total_start
    print(f"\n✅ 基准计算完成")
    print(f"   总耗时: {total_time:.2f}秒, 平均每个query: {total_time / len(queries):.4f}秒")
    if len(documents) > 0:
        print(f"   处理速度: {len(queries) * len(documents) / total_time:.0f} 次相似度计算/秒")

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
    output_file: Optional[str] = None,
    save_scores: bool = False
):
    """运行暴力法基准计算"""
    print("=" * 80)
    print("🚀 暴力法基准计算")
    print("=" * 80)

    # 1. 加载数据
    vector_data, model_name = load_vector_data(vector_file)
    metadata = vector_data.get('metadata', {})
    original_file = metadata.get('input_file', '')

    # 2. 提取向量
    try:
        queries, documents = extract_vectors(vector_data, original_file)
    except Exception as e:
        print(f"❌ 提取向量失败: {e}")
        return

    # 3. 计算基准
    print("\n" + "=" * 80)
    ground_truth, computation_times = calculate_benchmark(queries, documents, top_n)

    # 4. 计算统计信息
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

    # 5. 保存结果
    if output_file is None:
        output_file = vector_file.replace('.json', '_brute_force_benchmark.json')

    ground_truth_ids = [[doc_id for doc_id, _ in results] for results in ground_truth]

    results = {
        'vector_file': vector_file,
        'model': model_name,
        'query_count': len(queries),
        'document_count': len(documents),
        'top_n': top_n,
        'ground_truth': ground_truth_ids,
        'time_statistics': time_stats,
        'computation_times': [float(t) for t in computation_times],
        'metadata': {
            'queries': [{'query': q['query']} for q in queries]
        }
    }

    # 只在需要时保存分数（默认不保存以节省空间）
    if save_scores:
        results['ground_truth_with_scores'] = ground_truth
        print(f"   ℹ️  已保存相似度分数（文件可能较大）")
    else:
        print(f"   ℹ️  未保存相似度分数（使用 --save-scores 可启用）")

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
  python vector/vector_db/brute_force_benchmark.py -i .data/vectors/scidocs_gemini_vectors.json

  # 指定top-N
  python vector/vector_db/brute_force_benchmark.py -i .data/vectors/scidocs_gemini_vectors.json -n 20

  # 指定输出文件
  python vector/vector_db/brute_force_benchmark.py -i .data/vectors/scidocs_gemini_vectors.json -o benchmark_results.json
        """
    )

    parser.add_argument('-i', '--input', required=True, help='向量数据文件路径（JSON格式）')
    parser.add_argument('-n', '--top-n', type=int, default=10, help='Top-N检索数量（默认: 10）')
    parser.add_argument('-o', '--output', type=str, default=None, help='输出文件路径（默认: 自动生成）')
    parser.add_argument('--save-scores', action='store_true',
                       help='保存相似度分数（默认不保存以节省空间，10k queries × top-100 可能产生几百MB文件）')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        return

    try:
        run_brute_force_benchmark(
            vector_file=args.input,
            top_n=args.top_n,
            output_file=args.output,
            save_scores=args.save_scores
        )
    except KeyboardInterrupt:
        print("\n⚠️  用户中断")
    except Exception as e:
        print(f"\n❌ 计算失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
