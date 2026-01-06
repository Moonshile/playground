"""
Milvus向量数据库评测脚本

根据spec.md的要求，对Milvus进行向量检索评测：
- 基于向量化评测生成的q-d向量
- 所有document向量全部入库后再开始评测
- 评测基准：基于暴力法算出来的top-N最近邻向量
- 评测指标：
  - 检索能力：每个请求分别计算准确率和召回率，最后计算所有请求的平均值
  - 处理能力：记录每个请求的时间，并报告最终的时间分布（最大最小、平均、分位数）

使用方法:
    python vector/vector_db/milvus_benchmark.py -i .data/vectors/nfcorpus_openai_vectors.json
    python vector/vector_db/milvus_benchmark.py -i .data/vectors/nfcorpus_gemini_vec.json -n 10
"""
import os
import json
import time
import argparse
import hashlib
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    MilvusException
)


# ==================== 工具函数 ====================

def get_milvus_client():
    """获取Milvus客户端连接"""
    cluster_endpoint = os.getenv("MILVUS_CLUSTER_ENDPOINT")
    token = os.getenv("MILUVS_TOKEN") or os.getenv("MILVUS_TOKEN")  # 兼容拼写错误

    if not cluster_endpoint:
        raise ValueError("请设置 MILVUS_CLUSTER_ENDPOINT 环境变量")
    if not token:
        raise ValueError("请设置 MILVUS_TOKEN 或 MILUVS_TOKEN 环境变量")

    connections.connect(
        alias="default",
        uri=cluster_endpoint,
        token=token
    )
    return connections.get_connection_addr("default")


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


# ==================== Milvus操作 ====================

def create_collection(collection_name: str, dimension: int) -> Collection:
    """
    创建Milvus collection

    Args:
        collection_name: collection名称
        dimension: 向量维度

    Returns:
        Collection对象
    """
    # 检查collection是否已存在
    if utility.has_collection(collection_name):
        print(f"⚠️  Collection '{collection_name}' 已存在，删除旧collection...")
        collection = Collection(collection_name)
        collection.drop()

    # 定义schema
    fields = [
        FieldSchema(name="primary_key", dtype=DataType.VARCHAR, max_length=1024, is_primary=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension)
    ]

    schema = CollectionSchema(fields, description=f"Vector collection for {collection_name}")

    # 创建collection
    collection = Collection(collection_name, schema)

    # 创建索引
    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }
    collection.create_index("vector", index_params)

    print(f"✅ 已创建collection: {collection_name}")
    print(f"   维度: {dimension}")
    print(f"   索引类型: IVF_FLAT")

    return collection


def insert_documents(collection: Collection, documents: List[Dict[str, Any]], batch_size: int = 1000):
    """
    插入document向量到Milvus

    Args:
        collection: Milvus collection对象
        documents: document列表，每个包含hash和vector
        batch_size: 批量插入大小
    """
    print(f"📥 开始插入 {len(documents)} 个document向量...")

    primary_keys = [doc['hash'] for doc in documents]
    vectors = [doc['vector'] for doc in documents]

    # 批量插入
    total_batches = (len(documents) + batch_size - 1) // batch_size
    for i in range(0, len(documents), batch_size):
        batch_keys = primary_keys[i:i + batch_size]
        batch_vectors = vectors[i:i + batch_size]

        batch_num = i // batch_size + 1
        print(f"   插入批次 {batch_num}/{total_batches} ({len(batch_keys)} 条)...", end='\r')

        collection.insert([batch_keys, batch_vectors])

    # 刷新数据，确保可搜索
    collection.flush()
    print(f"\n✅ 已插入 {len(documents)} 个document向量")

    # 加载collection到内存
    collection.load()
    print("✅ Collection已加载到内存")


def search_vectors(
    collection: Collection,
    query_vectors: List[List[float]],
    top_k: int
) -> List[List[Dict[str, Any]]]:
    """
    在Milvus中搜索向量

    Args:
        collection: Milvus collection对象
        query_vectors: 查询向量列表
        top_k: 返回top-k结果

    Returns:
        每个query的搜索结果列表
    """
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

    results = collection.search(
        data=query_vectors,
        anns_field="vector",
        param=search_params,
        limit=top_k,
        output_fields=[]
    )

    # 转换结果格式
    formatted_results = []
    for result in results:
        hits = []
        for hit in result:
            hits.append({
                'id': hit.id,
                'distance': hit.distance,
                'score': hit.score if hasattr(hit, 'score') else hit.distance
            })
        formatted_results.append(hits)

    return formatted_results


# ==================== 评测指标 ====================

def calculate_metrics(
    retrieved_ids: List[str],
    ground_truth_ids: List[str]
) -> Tuple[float, float]:
    """
    计算准确率和召回率

    Args:
        retrieved_ids: 检索到的ID列表
        ground_truth_ids: 真实top-N的ID列表

    Returns:
        (准确率, 召回率)
    """
    retrieved_set = set(retrieved_ids)
    ground_truth_set = set(ground_truth_ids)

    # 交集
    intersection = retrieved_set & ground_truth_set

    # 准确率 = 检索结果中正确的数量 / 检索结果总数
    precision = len(intersection) / len(retrieved_ids) if retrieved_ids else 0.0

    # 召回率 = 检索结果中正确的数量 / 真实结果总数
    recall = len(intersection) / len(ground_truth_ids) if ground_truth_ids else 0.0

    return precision, recall


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


# ==================== 主评测流程 ====================

def load_ground_truth(ground_truth_file: str) -> Optional[List[List[str]]]:
    """加载预计算的基准结果"""
    if not os.path.exists(ground_truth_file):
        return None

    print(f"📖 正在加载基准结果: {ground_truth_file}")
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    ground_truth = data.get('ground_truth', [])
    if ground_truth:
        print(f"✅ 已加载 {len(ground_truth)} 个query的基准结果")
        return ground_truth

    return None


def run_benchmark(
    vector_file: str,
    top_n: int = 10,
    collection_name: Optional[str] = None,
    ground_truth_file: Optional[str] = None
):
    """
    运行Milvus评测

    Args:
        vector_file: 向量数据文件路径
        top_n: top-N检索数量
        collection_name: collection名称（如果为None，则根据模型自动生成）
    """
    print("=" * 80)
    print("🚀 Milvus向量数据库评测")
    print("=" * 80)

    # 1. 加载数据
    vector_data, model_name = load_vector_data(vector_file)
    results = vector_data.get('results', [])
    metadata = vector_data.get('metadata', {})

    # 尝试获取原始数据文件路径
    original_file = metadata.get('input_file', '')
    queries, documents = extract_query_document_vectors(results, original_file)

    if not queries:
        print("❌ 未找到query向量，无法进行评测")
        return

    if not documents:
        print("❌ 未找到document向量，无法进行评测")
        return

    # 去重documents（基于hash）
    unique_docs = {}
    for doc in documents:
        doc_hash = doc['hash']
        if doc_hash not in unique_docs:
            unique_docs[doc_hash] = doc

    documents = list(unique_docs.values())
    print(f"📊 去重后Document数量: {len(documents)}")

    # 2. 连接Milvus
    print("\n" + "=" * 80)
    print("🔌 连接Milvus...")
    try:
        get_milvus_client()
        print("✅ Milvus连接成功")
    except Exception as e:
        print(f"❌ Milvus连接失败: {e}")
        return

    # 3. 创建collection
    print("\n" + "=" * 80)
    print("📦 创建Collection...")
    if collection_name is None:
        # 根据模型名称生成collection名称
        model_safe = model_name.replace('@', '_').replace('/', '_').replace('-', '_')
        collection_name = f"benchmark_{model_safe}"

    # 获取向量维度
    dimension = len(documents[0]['vector'])
    collection = create_collection(collection_name, dimension)

    # 4. 插入所有document向量
    print("\n" + "=" * 80)
    print("📥 插入Document向量...")
    insert_documents(collection, documents)

    # 5. 计算或加载基准（暴力法）
    print("\n" + "=" * 80)
    print("🔍 计算基准（暴力法）...")

    ground_truth = None

    # 尝试加载预计算的基准
    if ground_truth_file:
        ground_truth = load_ground_truth(ground_truth_file)

    # 如果没有提供基准文件，尝试自动查找
    if ground_truth is None:
        possible_benchmark_file = vector_file.replace('.json', '_brute_force_benchmark.json')
        if os.path.exists(possible_benchmark_file):
            print(f"📖 发现基准文件: {possible_benchmark_file}")
            ground_truth = load_ground_truth(possible_benchmark_file)

    # 如果仍然没有基准，则计算
    if ground_truth is None:
        print(f"   正在为 {len(queries)} 个query计算top-{top_n}基准...")

        doc_vectors = [doc['vector'] for doc in documents]
        doc_ids = [doc['hash'] for doc in documents]

        ground_truth = []
        for i, query in enumerate(queries):
            if (i + 1) % 100 == 0:
                print(f"   进度: {i + 1}/{len(queries)}", end='\r')

            query_vec = query['vector']
            top_n_results = brute_force_top_n(query_vec, doc_vectors, doc_ids, top_n)
            ground_truth.append([doc_id for doc_id, _ in top_n_results])

        print(f"\n✅ 基准计算完成")
    else:
        # 验证基准数量是否匹配
        if len(ground_truth) != len(queries):
            print(f"⚠️  警告: 基准结果数量 ({len(ground_truth)}) 与query数量 ({len(queries)}) 不匹配")
            print(f"   将使用前 {min(len(ground_truth), len(queries))} 个结果")
            ground_truth = ground_truth[:len(queries)]

    # 6. 执行检索评测
    print("\n" + "=" * 80)
    print("🔎 执行检索评测...")

    precisions = []
    recalls = []
    search_times = []

    # 批量检索以提高效率
    batch_size = 10
    for i in range(0, len(queries), batch_size):
        batch_queries = queries[i:i + batch_size]
        batch_query_vectors = [q['vector'] for q in batch_queries]
        batch_ground_truth = ground_truth[i:i + batch_size]

        # 执行搜索
        search_start = time.time()
        batch_results = search_vectors(collection, batch_query_vectors, top_n)
        search_time = time.time() - search_start

        # 处理每个query的结果
        for j, (result, gt_ids) in enumerate(zip(batch_results, batch_ground_truth)):
            retrieved_ids = [hit['id'] for hit in result]

            # 计算指标
            precision, recall = calculate_metrics(retrieved_ids, gt_ids)
            precisions.append(precision)
            recalls.append(recall)

        # 平均每个query的搜索时间
        avg_time_per_query = search_time / len(batch_queries)
        search_times.extend([avg_time_per_query] * len(batch_queries))

        if (i + batch_size) % 100 == 0 or i + batch_size >= len(queries):
            print(f"   进度: {min(i + batch_size, len(queries))}/{len(queries)}", end='\r')

    print(f"\n✅ 检索评测完成")

    # 7. 计算最终指标
    print("\n" + "=" * 80)
    print("📊 评测结果")
    print("=" * 80)

    avg_precision = np.mean(precisions) if precisions else 0.0
    avg_recall = np.mean(recalls) if recalls else 0.0
    time_stats = calculate_time_statistics(search_times)

    print(f"\n🔍 检索能力:")
    print(f"   平均准确率: {avg_precision:.4f}")
    print(f"   平均召回率: {avg_recall:.4f}")

    print(f"\n⏱️  处理能力（每个请求的时间，单位：秒）:")
    if time_stats:
        print(f"   最小值: {time_stats['min']:.6f}")
        print(f"   最大值: {time_stats['max']:.6f}")
        print(f"   平均值: {time_stats['mean']:.6f}")
        print(f"   中位数: {time_stats['median']:.6f}")
        print(f"   P25: {time_stats['p25']:.6f}")
        print(f"   P75: {time_stats['p75']:.6f}")
        print(f"   P95: {time_stats['p95']:.6f}")
        print(f"   P99: {time_stats['p99']:.6f}")

    print(f"\n📈 统计信息:")
    print(f"   Query数量: {len(queries)}")
    print(f"   Document数量: {len(documents)}")
    print(f"   Top-N: {top_n}")
    print(f"   Collection: {collection_name}")

    # 保存结果
    results = {
        'vector_file': vector_file,
        'model': model_name,
        'collection_name': collection_name,
        'query_count': len(queries),
        'document_count': len(documents),
        'top_n': top_n,
        'metrics': {
            'average_precision': float(avg_precision),
            'average_recall': float(avg_recall),
            'precision_list': [float(p) for p in precisions],
            'recall_list': [float(r) for r in recalls]
        },
        'time_statistics': time_stats,
        'search_times': [float(t) for t in search_times]
    }

    output_file = vector_file.replace('.json', '_milvus_benchmark_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 结果已保存到: {output_file}")
    print("=" * 80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Milvus向量数据库评测脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本使用（会自动计算基准）
  python vector/vector_db/milvus_benchmark.py -i .data/vectors/nfcorpus_openai_vectors.json

  # 使用预计算的基准文件（推荐：先运行brute_force_benchmark.py生成基准）
  python vector/vector_db/milvus_benchmark.py -i .data/vectors/nfcorpus_openai_vectors.json -g benchmark_results.json

  # 指定top-N
  python vector/vector_db/milvus_benchmark.py -i .data/vectors/nfcorpus_gemini_vec.json -n 20

  # 指定collection名称
  python vector/vector_db/milvus_benchmark.py -i .data/vectors/nfcorpus_gemini_multimodal_vec.json -c my_collection
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
        '-c', '--collection',
        type=str,
        default=None,
        help='Collection名称（默认: 根据模型自动生成）'
    )

    parser.add_argument(
        '-g', '--ground-truth',
        type=str,
        default=None,
        help='预计算的基准结果文件路径（如果提供，将跳过基准计算）'
    )

    args = parser.parse_args()

    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        return

    try:
        run_benchmark(
            vector_file=args.input,
            top_n=args.top_n,
            collection_name=args.collection,
            ground_truth_file=args.ground_truth
        )
    except KeyboardInterrupt:
        print("\n⚠️  用户中断")
    except Exception as e:
        print(f"\n❌ 评测失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

