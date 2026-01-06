"""
将 mteb 检索数据集转换为 JSON 文件
输出格式：包含 query_list 和 document_list 的 JSON 对象，两个列表都已去重

输出格式示例:
{
  "query_list": ["查询文本1", "查询文本2", ...],
  "document_list": ["文档文本1", "文档文本2", ...]
}

使用方法:
    # 使用默认的小数据集 (nfcorpus)
    python vector/mteb/convert_to_json.py

    # 指定数据集和输出文件
    python vector/mteb/convert_to_json.py nfcorpus output.json

    # 指定拆分（train/test/validation）
    python vector/mteb/convert_to_json.py nfcorpus output.json --split train

    # 查看所有可用数据集
    python vector/mteb/convert_to_json.py --list
"""
from datasets import load_dataset, load_from_disk
from typing import Dict, Any, Optional, List
import sys
import os
import json
from pathlib import Path

# 复用 mteb_data_view.py 中的数据集列表和加载函数
SMALL_DATASETS = {
    "nfcorpus": {
        "name": "mteb/nfcorpus",
        "description": "NFCorpus - 约3,600个文档和323个查询，非常小",
        "size": "很小"
    },
    "scidocs": {
        "name": "mteb/scidocs",
        "description": "SciDocs - 科学文档检索数据集",
        "size": "小"
    },
    "scifact": {
        "name": "mteb/scifact",
        "description": "SciFact - 科学事实检索数据集",
        "size": "小"
    },
    "arguana": {
        "name": "mteb/arguana",
        "description": "ArguAna - 论证检索数据集",
        "size": "小"
    },
    "quora": {
        "name": "mteb/quora",
        "description": "Quora - Quora重复问题检测数据集",
        "size": "中等"
    },
    "msmarco": {
        "name": "mteb/msmarco",
        "description": "MS MARCO - 大规模检索数据集（较大）",
        "size": "很大"
    }
}

# 缓存文档查找结果
_doc_cache = {}
_query_cache = {}
# ID到文档的映射（预先构建，避免重复查找）
_doc_id_map = None
# ID到查询的映射（预先构建，避免重复查找）
_query_id_map = None

def get_cache_info():
    """获取缓存目录信息"""
    cache_dir = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    datasets_cache = os.path.join(cache_dir, "datasets")
    return datasets_cache

def load_dataset_with_cache(dataset_name: str, use_cache: bool = True) -> tuple:
    """加载数据集和corpus，使用缓存机制"""
    # 检查是否有本地保存的数据集
    local_path = f".data/{dataset_name.replace('/', '_')}"
    if os.path.exists(local_path) and use_cache:
        try:
            dataset = load_from_disk(local_path)
            print(f"✅ 从本地加载数据集: {local_path}")
            corpus = load_corpus_if_needed(dataset_name, use_cache)
            return dataset, corpus
        except Exception as e:
            print(f"⚠️  本地加载失败: {e}，将尝试从网络加载")

    # 从网络或缓存加载
    download_mode = None if use_cache else "force_redownload"
    try:
        dataset = load_dataset(dataset_name, download_mode=download_mode)
        print("✅ 数据集加载成功")
    except Exception as e:
        print(f"⚠️  加载失败: {e}")
        from datasets import get_dataset_config_names
        configs = get_dataset_config_names(dataset_name)
        if configs:
            dataset = load_dataset(dataset_name, configs[0], download_mode=download_mode)
            print(f"✅ 使用配置: {configs[0]}")
        else:
            raise

    corpus = load_corpus_if_needed(dataset_name, use_cache)
    return dataset, corpus

def load_queries_if_needed(dataset_name: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
    """尝试加载查询集合（queries）"""
    # 检查本地保存的queries
    queries_local_path = f".data/{dataset_name.replace('/', '_')}_queries"
    if os.path.exists(queries_local_path) and use_cache:
        try:
            queries = load_from_disk(queries_local_path)
            return queries
        except:
            pass

    # 提取基础名称
    base_name = dataset_name.replace("mteb/", "")
    queries_names = [
        f"{dataset_name}-queries",
        f"mteb/{base_name}-queries",
        f"{base_name}_queries",
    ]

    download_mode = None if use_cache else "force_redownload"

    for name in queries_names:
        try:
            queries = load_dataset(name, download_mode=download_mode)
            return queries
        except:
            continue

    # 尝试使用配置名
    try:
        queries = load_dataset(dataset_name, "queries", download_mode=download_mode)
        return queries
    except:
        pass

    return None

def load_corpus_if_needed(dataset_name: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
    """尝试加载文档集合（corpus）"""
    # 检查本地保存的corpus
    corpus_local_path = f".data/{dataset_name.replace('/', '_')}_corpus"
    if os.path.exists(corpus_local_path) and use_cache:
        try:
            corpus = load_from_disk(corpus_local_path)
            return corpus
        except:
            pass

    # 提取基础名称
    base_name = dataset_name.replace("mteb/", "")
    corpus_names = [
        f"{dataset_name}-corpus",
        f"mteb/{base_name}-corpus",
        base_name,
        f"{base_name}_corpus",
    ]

    download_mode = None if use_cache else "force_redownload"

    for name in corpus_names:
        try:
            corpus = load_dataset(name, download_mode=download_mode)
            return corpus
        except:
            continue

    # 尝试使用配置名
    try:
        corpus = load_dataset(dataset_name, "corpus", download_mode=download_mode)
        return corpus
    except:
        pass

    return None

def build_query_id_map(queries: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """预先构建ID到查询的映射，提高查找效率"""
    global _query_id_map

    if _query_id_map is not None:
        return _query_id_map

    _query_id_map = {}

    if queries is None:
        return _query_id_map

    print("正在构建查询ID映射（这只需要一次）...")

    for split_name in queries.keys():
        split_data = queries[split_name]
        if len(split_data) == 0:
            continue

        first_item = split_data[0]
        id_field = None
        text_field = None

        # 识别字段
        for key in first_item.keys():
            key_lower = key.lower()
            if 'id' in key_lower and id_field is None:
                id_field = key
            if ('text' in key_lower or 'content' in key_lower or
                'query' in key_lower):
                if text_field is None or 'text' in key_lower or 'query' in key_lower:
                    text_field = key

        if not text_field:
            for key in first_item.keys():
                value = first_item[key]
                if isinstance(value, str) and len(value) > 10:
                    text_field = key
                    break

        if text_field:
            # 批量构建映射
            total = len(split_data)
            for idx, item in enumerate(split_data):
                if id_field:
                    query_id = str(item.get(id_field))
                else:
                    query_id = str(idx)

                query_text = item.get(text_field)
                if query_text:
                    _query_id_map[query_id] = query_text

                if (idx + 1) % 1000 == 0:
                    print(f"  已处理查询: {idx + 1}/{total}")

    print(f"✅ 查询映射构建完成，共 {len(_query_id_map)} 个查询")
    return _query_id_map

def build_doc_id_map(corpus: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """预先构建ID到文档的映射，提高查找效率"""
    global _doc_id_map

    if _doc_id_map is not None:
        return _doc_id_map

    _doc_id_map = {}

    if corpus is None:
        return _doc_id_map

    print("正在构建文档ID映射（这只需要一次）...")

    for split_name in corpus.keys():
        split_data = corpus[split_name]
        if len(split_data) == 0:
            continue

        first_item = split_data[0]
        id_field = None
        text_field = None

        # 识别字段
        for key in first_item.keys():
            key_lower = key.lower()
            if 'id' in key_lower and id_field is None:
                id_field = key
            if ('text' in key_lower or 'content' in key_lower or
                'passage' in key_lower or 'body' in key_lower):
                if text_field is None or 'text' in key_lower or 'content' in key_lower:
                    text_field = key

        if not text_field:
            for key in first_item.keys():
                value = first_item[key]
                if isinstance(value, str) and len(value) > 50:
                    text_field = key
                    break

        if text_field:
            # 批量构建映射
            total = len(split_data)
            for idx, item in enumerate(split_data):
                if id_field:
                    doc_id = str(item.get(id_field))
                else:
                    # 如果没有ID字段，使用索引
                    doc_id = str(idx)

                doc_text = item.get(text_field)
                if doc_text:
                    _doc_id_map[doc_id] = doc_text

                if (idx + 1) % 10000 == 0:
                    print(f"  已处理文档: {idx + 1}/{total}")

    print(f"✅ 文档映射构建完成，共 {len(_doc_id_map)} 个文档")
    return _doc_id_map

def get_document_text(corpus: Optional[Dict[str, Any]], doc_id: Any) -> Optional[str]:
    """根据文档ID获取文档文本（使用预构建的映射）"""
    if corpus is None:
        return None

    # 使用预构建的映射
    doc_map = build_doc_id_map(corpus)
    cache_key = str(doc_id)

    # 先从映射中查找
    if cache_key in doc_map:
        return doc_map[cache_key]

    # 如果映射中没有，尝试从缓存中查找
    if cache_key in _doc_cache:
        return _doc_cache[cache_key]

    return None

def get_query_text(queries: Optional[Dict[str, Any]], query_id: Any, query_id_map: Optional[Dict[str, str]] = None) -> Optional[str]:
    """根据查询ID获取查询文本（使用预构建的映射）"""
    if queries is None:
        return None

    # 使用预构建的映射
    if query_id_map:
        cache_key = str(query_id)
        if cache_key in query_id_map:
            return query_id_map[cache_key]

    # 如果映射中没有，尝试从缓存中查找
    cache_key = str(query_id)
    if cache_key in _query_cache:
        return _query_cache[cache_key]

    return None

def is_likely_id(value: str) -> bool:
    """判断一个字符串是否可能是ID而不是实际文本"""
    if not isinstance(value, str):
        return True
    # ID通常较短，且不包含空格或标点符号（除了连字符、下划线）
    if len(value) < 3:
        return True
    if len(value) < 20 and not any(c in value for c in [' ', '.', ',', '?', '!', ':', ';']):
        # 可能是ID，但也要检查是否像文本
        # 如果包含常见单词，可能是文本
        common_words = ['the', 'and', 'or', 'is', 'are', 'was', 'were', 'what', 'how', 'why', 'when', 'where']
        value_lower = value.lower()
        if any(word in value_lower for word in common_words):
            return False
        # 如果全是数字或字母数字组合且很短，可能是ID
        if value.replace('-', '').replace('_', '').isalnum() and len(value) < 15:
            return True
    return False

def extract_query_and_document(example: Dict[str, Any], corpus: Optional[Dict[str, Any]], doc_id_map: Optional[Dict[str, str]] = None, queries: Optional[Dict[str, Any]] = None, query_id_map: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
    """从样本中提取query、document和score（优化版本）"""
    query = None
    document = None
    score = None
    query_id_value = None

    # 查找query字段
    for key, value in example.items():
        key_lower = key.lower()
        if 'query' in key_lower:
            if isinstance(value, str):
                # 如果已经是文本（长度较长或包含常见单词），直接使用
                if len(value) > 20 or not is_likely_id(value):
                    query = value
                else:
                    # 可能是ID，保存起来稍后查找
                    query_id_value = value
            else:
                query_id_value = str(value)
            break

    # 如果query是ID，尝试从queries数据集中查找
    if not query and query_id_value:
        if queries and query_id_map:
            query = get_query_text(queries, query_id_value, query_id_map)
        # 如果仍然找不到，且看起来像ID，返回None（跳过这个样本）
        if not query and is_likely_id(query_id_value):
            return None
        # 如果找不到但可能不是ID，使用原始值
        if not query:
            query = query_id_value

    if not query:
        return None

    # 优先查找已有的文档文本字段（避免ID查找）
    for key, value in example.items():
        key_lower = key.lower()
        if isinstance(value, str) and len(value) > 10:
            # 优先匹配包含文档内容的字段
            if any(x in key_lower for x in ['positive', 'passage', 'document', 'text', 'content', 'body']):
                if document is None or len(value) > len(document or ""):
                    document = value

    # 如果没有找到文档文本，尝试通过ID查找
    if not document and corpus and doc_id_map:
        for key, value in example.items():
            key_lower = key.lower()
            if ('id' in key_lower or 'passage' in key_lower or 'doc' in key_lower) and not ('query' in key_lower):
                if isinstance(value, (int, str)):
                    doc_id = str(value)
                    if doc_id in doc_id_map:
                        document = doc_id_map[doc_id]
                        break
                    # 如果ID在映射中找不到，且看起来像ID，跳过这个样本
                    elif is_likely_id(doc_id):
                        return None

    # 查找score字段
    # 优先级：1. 明确的score字段 2. positive/negative字段 3. relevance/rank/rating/label字段

    # 首先检查是否有positive/negative字段（检索数据集常用）
    has_positive = False
    has_negative = False
    for key, value in example.items():
        key_lower = key.lower()
        if 'positive' in key_lower and not 'negative' in key_lower:
            has_positive = True
            # 如果positive字段存在且是文档文本，说明这是正样本
            if isinstance(value, str) and len(value) > 10:
                score = 1.0
                break
        elif 'negative' in key_lower and not 'positive' in key_lower:
            has_negative = True
            # 如果negative字段存在且是文档文本，说明这是负样本
            if isinstance(value, str) and len(value) > 10:
                score = 0.0
                break

    # 如果没有通过positive/negative确定score，查找明确的score字段
    if score is None:
        for key, value in example.items():
            key_lower = key.lower()
            if 'score' in key_lower:
                # score可能是数字或字符串
                if isinstance(value, (int, float)):
                    score = float(value)
                elif isinstance(value, str):
                    try:
                        score = float(value)
                    except (ValueError, TypeError):
                        score = None
                else:
                    try:
                        score = float(value)
                    except (ValueError, TypeError):
                        score = None
                if score is not None:
                    break

    # 如果没有找到score字段，尝试查找其他可能的字段名
    if score is None:
        for key, value in example.items():
            key_lower = key.lower()
            # 可能的score相关字段名
            if any(x in key_lower for x in ['relevance', 'rank', 'rating', 'label']):
                if isinstance(value, (int, float)):
                    score = float(value)
                elif isinstance(value, str):
                    try:
                        score = float(value)
                    except (ValueError, TypeError):
                        pass
                else:
                    try:
                        score = float(value)
                    except (ValueError, TypeError):
                        pass
                if score is not None:
                    break

    # 如果仍然没有找到score，但数据集中有positive字段，默认设为1.0（正样本）
    # 如果有negative字段，默认设为0.0（负样本）
    if score is None:
        if has_positive:
            score = 1.0
        elif has_negative:
            score = 0.0

    # 最终验证：确保query和document都不是ID
    if query and document:
        # 如果query或document看起来还是ID，跳过
        if is_likely_id(query) or is_likely_id(document):
            return None
        result = {"query": query, "document": document}
        if score is not None:
            result["score"] = score
        return result
    return None

def convert_dataset_to_json(dataset_name: str, output_file: str, split: str = "train", use_cache: bool = True, reset_cache: bool = False) -> Dict[str, Any]:
    """将数据集转换为JSON格式"""
    print("=" * 60)
    print(f"正在转换数据集: {dataset_name}")
    print(f"输出文件: {output_file}")
    print(f"拆分: {split}")
    print("=" * 60)

    # 重置全局缓存（如果需要）
    global _doc_id_map, _doc_cache, _query_id_map, _query_cache
    if reset_cache:
        _doc_id_map = None
        _doc_cache = {}
        _query_id_map = None
        _query_cache = {}

    # 加载数据集
    cache_dir = get_cache_info()
    print(f"📦 缓存目录: {cache_dir}")
    print("正在加载数据集...")

    dataset, corpus = load_dataset_with_cache(dataset_name, use_cache)

    # 检查拆分是否存在
    if split not in dataset:
        available_splits = list(dataset.keys())
        print(f"❌ 拆分 '{split}' 不存在")
        print(f"可用的拆分: {available_splits}")
        if available_splits:
            split = available_splits[0]
            print(f"使用拆分: {split}")
        else:
            raise ValueError("数据集没有可用的拆分")

    split_data = dataset[split]
    print(f"\n📊 拆分信息:")
    print(f"  - 拆分名称: {split}")
    print(f"  - 样本数量: {len(split_data)}")
    fields = list(split_data[0].keys()) if len(split_data) > 0 else []
    print(f"  - 字段: {fields}")

    # 检查是否已经包含文档文本和查询文本（不需要corpus/queries）
    has_document_text = False
    has_query_text = False
    if len(split_data) > 0:
        first_item = split_data[0]
        for key, value in first_item.items():
            key_lower = key.lower()
            if isinstance(value, str) and len(value) > 10:
                if any(x in key_lower for x in ['positive', 'passage', 'document', 'text', 'content', 'body']):
                    has_document_text = True
                # 检查query字段是否已经是文本（而不是ID）
                if 'query' in key_lower and len(value) > 20:
                    has_query_text = True

    # 尝试加载queries数据集（只有在需要时）
    queries = None
    if use_cache and not has_query_text:
        print("\n正在尝试加载queries数据集...")
        queries = load_queries_if_needed(dataset_name, use_cache)
        if queries:
            print("✅ 成功加载queries数据集")
        else:
            print("⚠️  未找到queries数据集，将尝试从主数据集中提取")
    elif has_query_text:
        print("\n✅ 数据集中已包含查询文本，无需加载queries数据集")

    # 只有在需要时才加载和构建corpus映射
    doc_id_map = None
    if corpus and not has_document_text:
        print("\n检测到需要从corpus获取文档，正在构建映射...")
        doc_id_map = build_doc_id_map(corpus)
    elif has_document_text:
        print("\n✅ 数据集中已包含文档文本，无需加载corpus")

    # 预先构建查询映射（如果有queries）
    query_id_map = None
    if queries:
        query_id_map = build_query_id_map(queries)

    # 转换数据：收集所有唯一的 query 和 document
    print(f"\n正在转换数据...")
    query_set = set()  # 使用 set 自动去重
    document_set = set()  # 使用 set 自动去重
    skipped = 0
    total = len(split_data)

    # 使用批量处理提高效率
    batch_size = 1000
    for i in range(0, total, batch_size):
        batch_end = min(i + batch_size, total)
        batch = split_data.select(range(i, batch_end))

        for example in batch:
            result = extract_query_and_document(example, corpus, doc_id_map, queries, query_id_map)
            if result:
                query = result.get("query")
                document = result.get("document")
                if query:
                    query_set.add(query)
                if document:
                    document_set.add(document)
            else:
                skipped += 1

        # 更频繁的进度显示
        processed = batch_end
        progress = (processed / total) * 100
        print(f"  进度: {processed}/{total} ({progress:.1f}%) - 唯一query: {len(query_set)}, 唯一document: {len(document_set)}, 跳过: {skipped}")

    # 转换为列表并排序（保持顺序一致性）
    query_list = sorted(list(query_set))
    document_list = sorted(list(document_set))

    print(f"\n去重统计:")
    print(f"  原始样本数: {total}")
    print(f"  唯一query数: {len(query_list)}")
    print(f"  唯一document数: {len(document_list)}")
    print(f"  跳过样本数: {skipped}")

    # 验证转换结果，确保没有ID残留
    print(f"\n正在验证转换结果...")
    validation_issues = []
    id_like_queries = []
    id_like_docs = []

    for idx, query in enumerate(query_list):
        # 检查query是否还是ID
        if is_likely_id(query):
            id_like_queries.append({
                "index": idx,
                "query": query[:50] if len(query) > 50 else query
            })

    for idx, document in enumerate(document_list):
        # 检查document是否还是ID
        if is_likely_id(document):
            id_like_docs.append({
                "index": idx,
                "document": document[:50] if len(document) > 50 else document
            })

    # 报告验证结果
    if id_like_queries or id_like_docs:
        print(f"⚠️  警告：发现可能未转换的ID")
        if id_like_queries:
            print(f"  - 发现 {len(id_like_queries)} 个可能未转换的query ID")
            print(f"    前5个示例:")
            for issue in id_like_queries[:5]:
                print(f"      索引 {issue['index']}: {issue['query']}")
        if id_like_docs:
            print(f"  - 发现 {len(id_like_docs)} 个可能未转换的document ID")
            print(f"    前5个示例:")
            for issue in id_like_docs[:5]:
                print(f"      索引 {issue['index']}: {issue['document']}")

        validation_issues = {
            "query_ids": len(id_like_queries),
            "document_ids": len(id_like_docs),
            "total_issues": len(id_like_queries) + len(id_like_docs)
        }
    else:
        print("✅ 验证通过：所有ID都已转换为实际文本")
        validation_issues = {
            "query_ids": 0,
            "document_ids": 0,
            "total_issues": 0
        }

    # 保存JSON文件
    print(f"\n正在保存JSON文件...")
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)

    # 构建输出格式：{"query_list": [...], "document_list": [...]}
    output_data = {
        "query_list": query_list,
        "document_list": document_list
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # 统计信息
    stats = {
        "dataset": dataset_name,
        "split": split,
        "total_samples": len(split_data),
        "unique_queries": len(query_list),
        "unique_documents": len(document_list),
        "skipped_samples": skipped,
        "output_file": output_file,
        "file_size_mb": os.path.getsize(output_file) / (1024 * 1024),
        "validation": validation_issues
    }

    # 计算平均长度
    if query_list:
        avg_query_len = sum(len(q) for q in query_list) / len(query_list)
        stats["avg_query_length"] = round(avg_query_len, 2)

    if document_list:
        avg_doc_len = sum(len(d) for d in document_list) / len(document_list)
        stats["avg_document_length"] = round(avg_doc_len, 2)

    return stats

def print_dataset_list():
    """打印可用的数据集列表"""
    print("=" * 60)
    print("可用的检索数据集:")
    print("=" * 60)
    for key, info in SMALL_DATASETS.items():
        print(f"\n  {key}:")
        print(f"    - {info['description']}")
        print(f"    - 大小: {info['size']}")
        print(f"    - 数据集名: {info['name']}")

if __name__ == "__main__":
    dataset_key = "nfcorpus"
    output_file = "output.json"
    split = "train"
    use_cache = True

    # 解析命令行参数
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg in ["--list", "-l", "list"]:
            print_dataset_list()
            sys.exit(0)
        elif arg in ["--split", "-s"]:
            if i + 1 < len(sys.argv):
                split = sys.argv[i + 1]
                i += 2
            else:
                print("❌ --split 需要指定拆分名称")
                sys.exit(1)
        elif arg in ["--force-download", "-f"]:
            use_cache = False
            i += 1
        elif arg.startswith("--"):
            print(f"❌ 未知选项: {arg}")
            sys.exit(1)
        else:
            if i == 1:
                dataset_key = arg.lower()
            elif i == 2:
                output_file = arg
            i += 1

    # 获取数据集信息
    if dataset_key in SMALL_DATASETS:
        # 使用白名单中的数据集信息
        dataset_info = SMALL_DATASETS[dataset_key]
        dataset_name = dataset_info["name"]
        print(f"\n📦 使用数据集: {dataset_key}")
        print(f"   {dataset_info['description']}")
        print(f"   大小: {dataset_info['size']}\n")
    else:
        # 使用用户指定的数据集名称（自动添加 mteb/ 前缀如果不存在）
        if dataset_key.startswith("mteb/"):
            dataset_name = dataset_key
        else:
            dataset_name = f"mteb/{dataset_key}"
        print(f"\n📦 使用数据集: {dataset_name}")
        print(f"   (用户指定的数据集，不在推荐列表中)\n")

    # 转换数据
    try:
        stats = convert_dataset_to_json(dataset_name, output_file, split, use_cache)

        # 打印统计信息
        print("\n" + "=" * 60)
        print("转换完成！统计信息:")
        print("=" * 60)
        print(f"数据集: {stats['dataset']}")
        print(f"拆分: {stats['split']}")
        print(f"总样本数: {stats['total_samples']}")
        print(f"唯一query数: {stats['unique_queries']}")
        print(f"唯一document数: {stats['unique_documents']}")
        print(f"跳过样本: {stats['skipped_samples']}")
        if 'avg_query_length' in stats:
            print(f"平均查询长度: {stats['avg_query_length']} 字符")
        if 'avg_document_length' in stats:
            print(f"平均文档长度: {stats['avg_document_length']} 字符")
        print(f"输出文件: {stats['output_file']}")
        print(f"文件大小: {stats['file_size_mb']:.2f} MB")
        if 'validation' in stats:
            val = stats['validation']
            if val['total_issues'] > 0:
                print(f"\n⚠️  验证警告:")
                print(f"  可能未转换的query ID: {val['query_ids']}")
                print(f"  可能未转换的document ID: {val['document_ids']}")
            else:
                print(f"\n✅ 验证通过: 所有ID都已转换为实际文本")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

