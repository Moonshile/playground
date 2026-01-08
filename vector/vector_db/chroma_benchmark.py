"""
Chroma向量数据库评测脚本

根据spec.md的要求，对Chroma进行向量检索评测：
- 基于向量化评测生成的q-d向量
- 所有document向量全部入库后再开始评测
- 评测基准：基于暴力法算出来的top-N最近邻向量
- 评测指标：
  - 检索能力：每个请求分别计算准确率和召回率，最后计算所有请求的平均值
  - 处理能力：记录每个请求的时间，并报告最终的时间分布（最大最小、平均、分位数）

使用方法:
    python vector/vector_db/chroma_benchmark.py -i .data/vectors/scidocs_openai_vectors.json
    python vector/vector_db/chroma_benchmark.py -i .data/vectors/scidocs_gemini_vectors.json -n 10
"""
import os
import json
import time
import argparse
import hashlib
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from chromadb import CloudClient


# ==================== 工具函数 ====================

def get_chroma_client():
    """获取Chroma客户端连接（Cloud模式）"""
    api_key = os.getenv("CHROMA_API_KEY")
    tenant = os.getenv("CHROMA_TENANT")
    database = os.getenv("CHROMA_DATABASE")

    if not api_key:
        raise ValueError("请设置 CHROMA_API_KEY 环境变量")
    if not tenant:
        raise ValueError("请设置 CHROMA_TENANT 环境变量")
    if not database:
        raise ValueError("请设置 CHROMA_DATABASE 环境变量")

    # 创建Chroma Cloud客户端
    # 根据Chroma Cloud官方文档：https://docs.trychroma.com/docs/run-chroma/cloud-client
    # 使用CloudClient创建客户端
    try:
        client = CloudClient(
            tenant=tenant,
            database=database,
            api_key=api_key
        )
        print(f"✅ Chroma Cloud客户端创建成功")
        print(f"   Tenant: {tenant}")
        print(f"   Database: {database}")
    except Exception as e:
        error_msg = str(e)
        if "Permission denied" in error_msg or "permission" in error_msg.lower():
            raise ValueError(
                f"Chroma Cloud连接权限被拒绝。请检查：\n"
                f"  1. API密钥是否正确 (CHROMA_API_KEY)\n"
                f"  2. Tenant是否正确 (CHROMA_TENANT={tenant})\n"
                f"  3. Database是否正确 (CHROMA_DATABASE={database})\n"
                f"  4. API密钥是否有访问该tenant和database的权限\n"
                f"原始错误: {e}"
            )
        else:
            raise ValueError(f"创建Chroma Cloud客户端失败: {e}")

    return client


def compute_sha512_hex(text: str) -> str:
    """计算文本的SHA-512哈希值（十六进制字符串）"""
    return hashlib.sha512(text.encode('utf-8')).hexdigest()


# ==================== 数据加载 ====================

def load_vector_data(vector_file: str) -> Tuple[Dict[str, Any], str]:
    """
    加载向量数据文件

    Returns:
        (数据字典, 模型名称)
    """
    print(f"📖 正在加载向量数据: {vector_file}")
    with open(vector_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 获取模型名称
    model_name = data.get('metadata', {}).get('model', 'unknown')

    # 检查数据格式
    if 'query_vectors' in data and 'document_vectors' in data:
        # 新格式：分离的query_vectors和document_vectors
        query_count = len(data.get('query_vectors', []))
        doc_count = len(data.get('document_vectors', []))
        print(f"✅ 已加载数据（新格式）")
        print(f"   Query向量数量: {query_count}")
        print(f"   Document向量数量: {doc_count}")
        print(f"   模型: {model_name}")
    elif 'results' in data:
        # 旧格式：results列表
        results = data.get('results', [])
        print(f"✅ 已加载 {len(results)} 条数据（旧格式）")
        print(f"   模型: {model_name}")
    else:
        print(f"⚠️  未知的数据格式")

    return data, model_name


def load_original_data(original_file: str):
    """加载原始QA数据文件（支持新格式和旧格式）"""
    if not os.path.exists(original_file):
        return None

    print(f"📖 正在加载原始数据: {original_file}")
    with open(original_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict):
        # 新格式
        print(f"✅ 已加载原始数据（新格式）")
    elif isinstance(data, list):
        # 旧格式
        print(f"✅ 已加载 {len(data)} 条原始数据（旧格式）")
    else:
        print(f"⚠️  原始数据格式未知")

    return data


def extract_query_document_vectors_new_format(
    vector_data: Dict[str, Any],
    original_data_file: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    从新格式的向量数据中提取query和document向量

    Args:
        vector_data: 向量数据字典（包含query_vectors和document_vectors）
        original_data_file: 原始QA数据文件路径

    Returns:
        (query列表, document列表)
    """
    queries = []
    documents = []

    query_vectors = vector_data.get('query_vectors', [])
    document_vectors = vector_data.get('document_vectors', [])

    # 加载原始数据
    original_data = load_original_data(original_data_file)
    if not original_data:
        print("❌ 无法加载原始数据文件")
        return queries, documents

    # 检查原始数据格式
    if isinstance(original_data, dict):
        # 新格式：包含query_list和document_list
        if 'query_list' in original_data and 'document_list' in original_data:
            query_list = original_data['query_list']
            document_list = original_data['document_list']
        else:
            print("❌ 原始数据格式不正确（新格式应包含query_list和document_list）")
            return queries, documents
    elif isinstance(original_data, list):
        # 旧格式：列表，每个item包含query和document
        # 提取唯一的query和document列表（按首次出现的顺序）
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
    else:
        print("❌ 原始数据格式未知")
        return queries, documents

    # 匹配向量和文本
    print(f"📊 匹配向量和文本...")
    print(f"   唯一Query: {len(query_list)}")
    print(f"   唯一Document: {len(document_list)}")
    print(f"   Query向量: {len(query_vectors)}")
    print(f"   Document向量: {len(document_vectors)}")

    # 验证数量
    if len(query_vectors) != len(query_list):
        print(f"⚠️  警告: Query向量数量 ({len(query_vectors)}) 与唯一Query数量 ({len(query_list)}) 不匹配")

    if len(document_vectors) != len(document_list):
        print(f"⚠️  警告: Document向量数量 ({len(document_vectors)}) 与唯一Document数量 ({len(document_list)}) 不匹配")

    # 匹配query向量（按顺序）
    for i, query_text in enumerate(query_list):
        if i < len(query_vectors):
            queries.append({
                'query': query_text,
                'vector': query_vectors[i],
                'document': '',  # 新格式不包含document关联
                'score': None
            })

    # 匹配document向量（按顺序）
    for i, doc_text in enumerate(document_list):
        if i < len(document_vectors):
            doc_hash = compute_sha512_hex(doc_text)
            documents.append({
                'document': doc_text,
                'vector': document_vectors[i],
                'hash': doc_hash
            })

    return queries, documents


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
            doc_hash = compute_sha512_hex(doc_text)
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
                doc_hash = compute_sha512_hex(doc_text)
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


# ==================== Chroma操作 ====================

def create_collection(client, collection_name: str, dimension: int, force_recreate: bool = False):
    """
    创建或获取Chroma collection

    Args:
        client: Chroma客户端
        collection_name: collection名称
        dimension: 向量维度（Chroma会自动推断，但我们可以验证）
        force_recreate: 是否强制重新创建（删除已存在的collection）

    Returns:
        (Collection对象, 是否是新创建的)
    """
    # 检查collection是否已存在
    try:
        existing_collection = client.get_collection(collection_name)
        if force_recreate:
            print(f"⚠️  Collection '{collection_name}' 已存在，强制删除旧collection...")
            client.delete_collection(collection_name)
        else:
            print(f"✅ Collection '{collection_name}' 已存在，复用现有collection")
            # 验证维度
            metadata = existing_collection.metadata or {}
            existing_dim = metadata.get('dimension')
            if existing_dim and existing_dim != dimension:
                print(f"⚠️  警告: 现有collection的维度 ({existing_dim}) 与预期维度 ({dimension}) 不匹配")
            return existing_collection, False
    except Exception:
        # Collection不存在，继续创建
        pass

    # 创建新collection
    # Chroma会自动推断向量维度，但我们可以在metadata中存储
    collection = client.create_collection(
        name=collection_name,
        metadata={"dimension": dimension}
    )

    print(f"✅ 已创建collection: {collection_name}")
    print(f"   维度: {dimension}")

    return collection, True


def check_collection_data(collection, expected_count: int) -> bool:
    """
    检查collection中的数据量是否匹配预期

    Args:
        collection: Chroma collection对象
        expected_count: 预期的数据量

    Returns:
        是否匹配
    """
    try:
        # 获取collection的count
        count = collection.count()
        return count == expected_count
    except Exception as e:
        print(f"   ⚠️  检查collection数据量时出错: {e}")
        return False


def insert_documents(collection, documents: List[Dict[str, Any]], batch_size: int = 100, skip_if_exists: bool = True):
    """
    插入document向量到Chroma

    Args:
        collection: Chroma collection对象
        documents: document列表，每个包含hash和vector
        batch_size: 批量插入大小
        skip_if_exists: 如果collection中已有数据，是否跳过插入
    """
    # 检查是否已有数据
    if skip_if_exists:
        try:
            current_count = collection.count()
            expected_count = len(documents)
            print(f"📊 Collection当前数据量: {current_count}，预期插入: {expected_count}")

            if current_count >= expected_count:
                print(f"✅ Collection中已有 {current_count} 条数据（预期 {expected_count} 条），跳过插入")
                return
            elif current_count > 0:
                print(f"⚠️  Collection中已有 {current_count} 条数据，但预期 {expected_count} 条")
                print(f"   检查哪些数据已存在...")

                # 分批检查已存在的ID，避免一次性查询太多
                all_ids = [doc['hash'] for doc in documents]
                existing_ids = set()
                check_batch_size = 100  # 每次检查100个ID

                for i in range(0, len(all_ids), check_batch_size):
                    batch_ids = all_ids[i:i + check_batch_size]
                    try:
                        existing_results = collection.get(ids=batch_ids)
                        batch_existing = existing_results.get('ids', [])
                        existing_ids.update(batch_existing)
                    except Exception as e:
                        # 如果检查失败，保守处理：假设这些ID已存在，避免重复插入
                        print(f"   ⚠️  检查批次 {i//check_batch_size + 1} 时出错: {e}")
                        print(f"   保守处理：假设这些ID已存在，跳过插入")
                        existing_ids.update(batch_ids)

                # 过滤出需要插入的数据
                documents = [doc for doc in documents if doc['hash'] not in existing_ids]

                if not documents:
                    print(f"✅ 所有数据已存在，跳过插入")
                    return

                print(f"   实际需要插入 {len(documents)} 条新数据（已存在 {len(existing_ids)} 条）")
        except Exception as e:
            print(f"   ⚠️  检查collection数据量时出错: {e}")
            print(f"   ⚠️  为避免配额超限，将跳过插入。如需强制插入，请使用 --force-recreate")
            # 如果检查失败，为了安全起见，不插入数据
            raise ValueError(
                f"无法检查collection数据状态，为避免配额超限已跳过插入。\n"
                f"如果确定需要插入，请使用 --force-recreate 参数强制重建collection。\n"
                f"原始错误: {e}"
            )

    print(f"📥 开始插入 {len(documents)} 个document向量...")

    ids = [doc['hash'] for doc in documents]
    embeddings = [doc['vector'] for doc in documents]
    # Chroma需要metadatas，我们可以存储document文本（可选）
    metadatas = [{"text": doc['document'][:1000]} for doc in documents]  # 限制长度

    # 批量插入
    total_batches = (len(documents) + batch_size - 1) // batch_size
    inserted_count = 0

    for i in range(0, len(documents), batch_size):
        batch_ids = ids[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]
        batch_metadatas = metadatas[i:i + batch_size]

        batch_num = i // batch_size + 1
        print(f"   插入批次 {batch_num}/{total_batches} ({len(batch_ids)} 条)...", end='\r')

        try:
            collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas
            )
            inserted_count += len(batch_ids)
        except Exception as e:
            error_msg = str(e)
            if "Quota exceeded" in error_msg or "quota" in error_msg.lower():
                print(f"\n❌ Chroma Cloud配额超限错误:")
                print(f"   {error_msg}")
                print(f"\n📊 插入进度: 已成功插入 {inserted_count}/{len(documents)} 条数据")
                print(f"\n💡 解决方案:")
                print(f"   1. 当前collection可能已有数据，请检查并考虑使用 --force-recreate 强制重建")
                print(f"   2. 联系Chroma Cloud申请增加配额（错误信息中包含申请链接）")
                print(f"   3. 使用较小的数据集进行测试")
                print(f"   4. 如果数据已部分插入，可以继续运行脚本，脚本会跳过已存在的数据")
                raise ValueError(f"Chroma Cloud配额超限: {error_msg}")
            else:
                # 其他错误继续抛出
                raise

    print(f"\n✅ 已插入 {inserted_count} 个document向量")


def search_vectors(
    collection,
    query_vectors: List[List[float]],
    top_k: int
) -> List[List[Dict[str, Any]]]:
    """
    在Chroma中搜索向量

    Args:
        collection: Chroma collection对象
        query_vectors: 查询向量列表
        top_k: 返回top-k结果

    Returns:
        每个query的搜索结果列表
    """
    # Chroma支持批量查询
    results = collection.query(
        query_embeddings=query_vectors,
        n_results=top_k
    )

    # 转换结果格式
    formatted_results = []
    # results的结构: {'ids': [[id1, id2, ...], ...], 'distances': [[dist1, dist2, ...], ...], ...}
    num_queries = len(query_vectors)
    for i in range(num_queries):
        hits = []
        query_ids = results['ids'][i] if i < len(results['ids']) else []
        query_distances = results['distances'][i] if i < len(results['distances']) else []

        for doc_id, distance in zip(query_ids, query_distances):
            hits.append({
                'id': doc_id,
                'distance': float(distance),
                'score': float(distance)  # Chroma使用距离，我们也可以转换为相似度
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

def find_benchmark_file(vector_file: str) -> Optional[str]:
    """
    查找基准文件（优先从.data/vector_search目录）

    Args:
        vector_file: 向量数据文件路径

    Returns:
        基准文件路径，如果找不到则返回None
    """
    # 从向量文件名生成基准文件名
    base_name = os.path.basename(vector_file)
    benchmark_name = base_name.replace('.json', '_brute_force_benchmark.json')

    # 优先查找路径（按优先级）
    possible_paths = [
        # 1. .data/vector_search目录（优先）
        os.path.join('.data', 'vector_search', benchmark_name),
        # 2. 与向量文件同目录
        vector_file.replace('.json', '_brute_force_benchmark.json'),
        # 3. 向量文件所在目录
        os.path.join(os.path.dirname(vector_file), benchmark_name),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    return None


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
        # 验证基准文件信息
        benchmark_top_n = data.get('top_n', 0)
        benchmark_query_count = data.get('query_count', 0)
        print(f"   基准信息: top_n={benchmark_top_n}, query_count={benchmark_query_count}")
        return ground_truth

    return None


def run_benchmark(
    vector_file: str,
    top_n: int = 10,
    collection_name: Optional[str] = None,
    ground_truth_file: Optional[str] = None,
    force_recreate: bool = False
):
    """
    运行Chroma评测

    Args:
        vector_file: 向量数据文件路径
        top_n: top-N检索数量
        collection_name: collection名称（如果为None，则根据模型自动生成）
        ground_truth_file: 预计算的基准结果文件路径
        force_recreate: 是否强制重新创建collection
    """
    print("=" * 80)
    print("🚀 Chroma向量数据库评测")
    print("=" * 80)

    # 1. 加载数据
    vector_data, model_name = load_vector_data(vector_file)
    metadata = vector_data.get('metadata', {})
    original_file = metadata.get('input_file', '')

    # 检查数据格式
    if 'query_vectors' in vector_data and 'document_vectors' in vector_data:
        # 新格式：分离的query_vectors和document_vectors
        print(f"\n检测到新格式数据，使用新格式解析...")
        if not original_file:
            print("❌ 新格式需要原始数据文件路径，但metadata中未找到input_file")
            return

        # 尝试多个可能的路径
        possible_paths = [
            original_file,
            os.path.join('.data', 'mteb', os.path.basename(original_file)),
            os.path.join('.data/mteb', os.path.basename(original_file))
        ]

        found_original_file = None
        for path in possible_paths:
            if os.path.exists(path):
                found_original_file = path
                break

        if not found_original_file:
            print(f"❌ 无法找到原始数据文件: {original_file}")
            return

        queries, documents = extract_query_document_vectors_new_format(vector_data, found_original_file)
    else:
        # 旧格式：results列表
        results = vector_data.get('results', [])
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

    # 2. 连接Chroma
    print("\n" + "=" * 80)
    print("🔌 连接Chroma...")
    try:
        client = get_chroma_client()
        print("✅ Chroma连接成功")
    except Exception as e:
        print(f"❌ Chroma连接失败: {e}")
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
    collection, is_new_collection = create_collection(client, collection_name, dimension, force_recreate)

    # 4. 插入所有document向量（如果是新collection或强制重建，则插入；否则检查是否已有数据）
    print("\n" + "=" * 80)
    print("📥 插入Document向量...")
    insert_documents(collection, documents, skip_if_exists=not force_recreate)

    # 5. 加载基准（优先从.data/vector_search目录）
    print("\n" + "=" * 80)
    print("🔍 加载基准结果...")

    ground_truth = None
    benchmark_file_used = None

    # 优先使用用户指定的基准文件
    if ground_truth_file:
        if os.path.exists(ground_truth_file):
            ground_truth = load_ground_truth(ground_truth_file)
            benchmark_file_used = ground_truth_file
        else:
            print(f"⚠️  指定的基准文件不存在: {ground_truth_file}")

    # 如果没有指定或文件不存在，自动查找基准文件
    if ground_truth is None:
        benchmark_file = find_benchmark_file(vector_file)
        if benchmark_file:
            print(f"📖 自动发现基准文件: {benchmark_file}")
            ground_truth = load_ground_truth(benchmark_file)
            benchmark_file_used = benchmark_file

    # 如果仍然没有基准，报错（不再自动计算）
    if ground_truth is None:
        print(f"❌ 未找到基准文件，无法进行评测")
        print(f"   请先运行 brute_force_benchmark.py 生成基准文件")
        print(f"   或使用 -g 参数指定基准文件路径")
        print(f"   预期基准文件位置:")
        print(f"     - .data/vector_search/{os.path.basename(vector_file).replace('.json', '_brute_force_benchmark.json')}")
        print(f"     - {vector_file.replace('.json', '_brute_force_benchmark.json')}")
        return

    # 验证基准文件信息
    if benchmark_file_used:
        with open(benchmark_file_used, 'r', encoding='utf-8') as f:
            benchmark_data = json.load(f)
        benchmark_top_n = benchmark_data.get('top_n', 0)
        benchmark_query_count = benchmark_data.get('query_count', 0)

        # 验证top_n是否匹配
        if benchmark_top_n != top_n:
            print(f"⚠️  警告: 基准文件的top_n ({benchmark_top_n}) 与指定的top_n ({top_n}) 不匹配")
            print(f"   将使用基准文件的top_n: {benchmark_top_n}")
            top_n = benchmark_top_n

        # 验证query数量是否匹配
        if len(ground_truth) != len(queries):
            print(f"⚠️  警告: 基准结果数量 ({len(ground_truth)}) 与query数量 ({len(queries)}) 不匹配")
            if len(ground_truth) < len(queries):
                print(f"   基准结果不足，无法完成评测")
                return
            else:
                print(f"   将使用前 {len(queries)} 个结果")
                ground_truth = ground_truth[:len(queries)]

    print(f"✅ 基准加载完成，使用文件: {benchmark_file_used}")
    print(f"   Top-N: {top_n}, Query数量: {len(queries)}")

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

    output_file = vector_file.replace('.json', '_chroma_benchmark_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 结果已保存到: {output_file}")
    print("=" * 80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Chroma向量数据库评测脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本使用（自动从.data/vector_search目录加载基准文件）
  python vector/vector_db/chroma_benchmark.py -i .data/vectors/scidocs_openai_vectors.json

  # 指定基准文件路径（如果不在默认位置）
  python vector/vector_db/chroma_benchmark.py -i .data/vectors/scidocs_openai_vectors.json -g path/to/benchmark.json

  # 指定top-N（会自动从基准文件读取，如果基准文件的top_n不同会给出警告）
  python vector/vector_db/chroma_benchmark.py -i .data/vectors/scidocs_gemini_vectors.json -n 20

  # 指定collection名称
  python vector/vector_db/chroma_benchmark.py -i .data/vectors/scidocs_openai_vectors.json -c my_collection

  # 强制重新创建collection（删除已存在的并重新插入数据）
  python vector/vector_db/chroma_benchmark.py -i .data/vectors/scidocs_openai_vectors.json --force-recreate

环境变量:
  - CHROMA_API_KEY: Chroma Cloud API密钥
  - CHROMA_TENANT: Chroma Cloud租户名称
  - CHROMA_DATABASE: Chroma Cloud数据库名称

注意:
  - 基准文件会自动从.data/vector_search目录查找（优先）
  - 如果找不到基准文件，评测会失败并提示需要先运行brute_force_benchmark.py
  - 基准文件应包含ground_truth字段（query的top-N结果列表）
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

    parser.add_argument(
        '--force-recreate',
        action='store_true',
        help='强制重新创建collection（删除已存在的collection并重新插入数据）'
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
            ground_truth_file=args.ground_truth,
            force_recreate=args.force_recreate
        )
    except KeyboardInterrupt:
        print("\n⚠️  用户中断")
    except Exception as e:
        print(f"\n❌ 评测失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

