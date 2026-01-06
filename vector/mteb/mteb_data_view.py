"""
使用 datasets 库查看 mteb 中的检索数据集内容示例
支持多种数据集，包括较小的替代选项

使用方法:
    # 使用默认的小数据集 (nfcorpus) - 会自动使用缓存
    python vector/mteb/mteb_data_view.py

    # 指定数据集
    python vector/mteb/mteb_data_view.py nfcorpus
    python vector/mteb/mteb_data_view.py scidocs

    # 保存到本地以便下次更快加载（推荐）
    python vector/mteb/mteb_data_view.py nfcorpus --save-local

    # 强制重新下载（忽略缓存）
    python vector/mteb/mteb_data_view.py nfcorpus --force-download

    # 查看所有可用数据集
    python vector/mteb/mteb_data_view.py --list

缓存说明:
    - datasets 库默认会缓存下载的数据集到 ~/.cache/huggingface/datasets
    - 首次运行会下载数据集，后续运行会直接使用缓存，不会重新下载
    - 使用 --save-local 可以将数据集保存到项目本地 .data/ 目录，加载更快
    - 使用 --force-download 可以强制重新下载（忽略缓存）

推荐的小型数据集:
    - nfcorpus: 最小，约3,600个文档和323个查询
    - scidocs: 科学文档检索
    - scifact: 科学事实检索
    - arguana: 论证检索
"""
from datasets import load_dataset, load_from_disk
from typing import Dict, Any, Optional
import sys
import os
from pathlib import Path

# 推荐的小型检索数据集列表
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

def load_corpus_if_needed(dataset_name: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
    """尝试加载文档集合（corpus）"""
    # 检查本地保存的corpus
    corpus_local_path = f".data/{dataset_name.replace('/', '_')}_corpus"
    if os.path.exists(corpus_local_path) and use_cache:
        try:
            corpus = load_from_disk(corpus_local_path)
            print(f"✅ 从本地加载corpus: {corpus_local_path}")
            return corpus
        except:
            pass

    # 提取基础名称（去掉mteb/前缀）
    base_name = dataset_name.replace("mteb/", "")

    # 尝试多种可能的corpus数据集名称
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
            print(f"✅ 成功加载文档集合: {name}")
            return corpus
        except Exception as e:
            continue

    # 尝试使用配置名
    try:
        corpus = load_dataset(dataset_name, "corpus", download_mode=download_mode)
        print("✅ 成功加载corpus数据集（使用配置名）")
        return corpus
    except:
        pass

    # 尝试直接加载可能包含文档的数据集
    try:
        # 检查是否有其他配置
        from datasets import get_dataset_config_names
        configs = get_dataset_config_names(dataset_name)
        print(f"📋 可用的配置: {configs}")
        for config in configs:
            if 'corpus' in config.lower() or 'passage' in config.lower() or 'doc' in config.lower():
                corpus = load_dataset(dataset_name, config, download_mode=download_mode)
                print(f"✅ 成功加载配置: {config}")
                return corpus
    except:
        pass

    print("⚠️  无法加载corpus数据集，将只显示ID")
    return None

# 缓存文档查找结果
_doc_cache = {}

def get_document_text(corpus: Optional[Dict[str, Any]], doc_id: Any) -> Optional[str]:
    """根据文档ID获取文档文本"""
    if corpus is None:
        return None

    # 使用缓存
    cache_key = str(doc_id)
    if cache_key in _doc_cache:
        return _doc_cache[cache_key]

    # 尝试在不同拆分中查找
    for split_name in corpus.keys():
        split_data = corpus[split_name]
        if len(split_data) == 0:
            continue

        # 检查字段结构
        first_item = split_data[0]
        id_field = None
        text_field = None

        # 查找ID字段和文本字段
        for key in first_item.keys():
            key_lower = key.lower()
            if 'id' in key_lower and id_field is None:
                id_field = key
            if ('text' in key_lower or 'content' in key_lower or
                'passage' in key_lower or 'body' in key_lower or
                ('title' in key_lower and text_field is None)):
                if text_field is None or 'text' in key_lower or 'content' in key_lower:
                    text_field = key

        if not text_field:
            # 如果没有找到明确的文本字段，尝试使用最长的字符串字段
            for key in first_item.keys():
                value = first_item[key]
                if isinstance(value, str) and len(value) > 50:
                    text_field = key
                    break

        if text_field:
            # 尝试查找文档
            try:
                # 如果doc_id是数字，尝试直接索引
                if isinstance(doc_id, (int, str)) and str(doc_id).isdigit():
                    idx = int(doc_id)
                    if 0 <= idx < len(split_data):
                        result = split_data[idx].get(text_field)
                        if result:
                            _doc_cache[cache_key] = result
                            return result

                # 如果有ID字段，通过ID查找
                if id_field:
                    for item in split_data:
                        if str(item.get(id_field)) == str(doc_id):
                            result = item.get(text_field)
                            if result:
                                _doc_cache[cache_key] = result
                                return result

                # 如果ID字段就是索引，直接使用
                if id_field and split_data[0].get(id_field) == 0:
                    # 可能是索引字段
                    try:
                        idx = int(doc_id)
                        if 0 <= idx < len(split_data):
                            result = split_data[idx].get(text_field)
                            if result:
                                _doc_cache[cache_key] = result
                                return result
                    except:
                        pass
            except Exception as e:
                pass

    return None

def get_cache_info():
    """获取缓存目录信息"""
    cache_dir = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    datasets_cache = os.path.join(cache_dir, "datasets")
    return datasets_cache

def view_dataset_data(dataset_name: str = "mteb/nfcorpus", use_cache: bool = True, save_local: bool = False):
    """加载并查看指定数据集的内容

    Args:
        dataset_name: 数据集名称
        use_cache: 是否使用缓存（默认True，会使用已下载的数据）
        save_local: 是否保存到本地目录以便更快加载（默认False）
    """

    print("=" * 60)
    print(f"正在加载数据集: {dataset_name}")
    print("=" * 60)

    # 显示缓存信息
    cache_dir = get_cache_info()
    print(f"📦 缓存目录: {cache_dir}")
    if use_cache:
        print("✅ 将使用缓存（如果已下载，不会重新下载）")
    else:
        print("⚠️  将强制重新下载（忽略缓存）")

    # 检查是否有本地保存的数据集
    local_path = f".data/{dataset_name.replace('/', '_')}"
    dataset = None
    if os.path.exists(local_path) and use_cache:
        print(f"\n📂 发现本地保存的数据集: {local_path}")
        print("   正在从本地加载（最快）...")
        try:
            dataset = load_from_disk(local_path)
            print("✅ 成功从本地加载数据集")
        except Exception as e:
            print(f"⚠️  本地加载失败: {e}，将尝试从网络加载")
            dataset = None

    # 如果本地加载失败或未找到，从网络加载
    if dataset is None:
        # 加载数据集 - 尝试不同的配置
        print("\n正在从网络加载数据集...")
        try:
            # 如果设置了use_cache=False，可以通过设置download_mode控制
            download_mode = None if use_cache else "force_redownload"
            dataset = load_dataset(dataset_name, download_mode=download_mode)
            print("✅ 数据集加载成功")

            # 如果启用保存本地，保存数据集
            if save_local:
                os.makedirs(local_path, exist_ok=True)
                print(f"\n💾 正在保存数据集到本地: {local_path}")
                dataset.save_to_disk(local_path)
                print("✅ 数据集已保存到本地，下次运行将更快加载")

        except Exception as e:
            print(f"⚠️  加载失败: {e}")
            # 尝试查看可用配置
            try:
                from datasets import get_dataset_config_names
                configs = get_dataset_config_names(dataset_name)
                print(f"📋 可用的配置: {configs}")
                if configs:
                    download_mode = None if use_cache else "force_redownload"
                    dataset = load_dataset(dataset_name, configs[0], download_mode=download_mode)
                    print(f"✅ 使用配置: {configs[0]}")
                    if save_local:
                        os.makedirs(local_path, exist_ok=True)
                        dataset.save_to_disk(local_path)
            except:
                raise

    # 尝试加载文档集合
    print("\n正在尝试加载文档集合...")
    corpus = load_corpus_if_needed(dataset_name, use_cache=use_cache)

    # 查看数据集基本信息
    print("\n📊 数据集基本信息:")
    print(dataset)

    # 查看各个拆分的详细信息
    print("\n" + "=" * 60)
    print("数据集拆分详情:")
    print("=" * 60)

    for split_name in dataset.keys():
        split_data = dataset[split_name]
        print(f"\n{split_name.upper()} 拆分:")
        print(f"  - 样本数量: {len(split_data)}")
        print(f"  - 特征字段: {split_data.features}")

        # 获取第一个样本来检查实际字段名
        if len(split_data) > 0:
            first_item = split_data[0]
            print(f"  - 实际字段名: {list(first_item.keys())}")

        # 显示前几个样本
        print(f"\n  前 3 个样本示例:")
        print("-" * 60)
        for i, example in enumerate(split_data.select(range(min(3, len(split_data))))):
            print(f"\n  样本 {i+1}:")
            # 显示所有字段，并尝试获取文档内容
            for key, value in example.items():
                # 如果值看起来像ID（数字或短字符串）且字段名包含id/passage/doc，尝试获取实际文本
                is_id_field = ('id' in key.lower() or 'passage' in key.lower() or 'doc' in key.lower())
                is_id_value = (isinstance(value, (int, str)) and
                              (isinstance(value, int) or (isinstance(value, str) and len(str(value)) < 50 and not ' ' in str(value))))

                if corpus and is_id_field and is_id_value:
                    doc_text = get_document_text(corpus, value)
                    if doc_text:
                        print(f"    {key} (ID): {value}")
                        print(f"    {key}_text: {doc_text[:300]}..." if len(doc_text) > 300 else f"    {key}_text: {doc_text}")
                    else:
                        print(f"    {key}: {value} (未找到对应文档)")
                elif isinstance(value, str):
                    # 如果已经是文本内容，直接显示
                    if len(value) > 300:
                        print(f"    {key}: {value[:300]}...")
                    else:
                        print(f"    {key}: {value}")
                else:
                    print(f"    {key}: {value}")

    # 查看完整样本（第一个）
    print("\n" + "=" * 60)
    # 尝试找到第一个可用的拆分
    first_split_name = None
    first_split_data = None
    for split_name in dataset.keys():
        if len(dataset[split_name]) > 0:
            first_split_name = split_name
            first_split_data = dataset[split_name]
            break

    if first_split_name:
        print(f"完整样本示例 ({first_split_name} 拆分第一个样本):")
        print("=" * 60)
        first_example = first_split_data[0]
        print(f"\n完整样本内容:")
        for key, value in first_example.items():
            print(f"\n{key}:")
            # 如果是ID字段，尝试获取实际文本
            if corpus and ('id' in key.lower() or 'passage' in key.lower() or 'doc' in key.lower()):
                doc_text = get_document_text(corpus, value)
                if doc_text:
                    print(f"  ID: {value}")
                    print(f"  实际内容: {doc_text}")
                else:
                    print(f"  {value}")
            else:
                # 如果是长文本，截断显示
                if isinstance(value, str) and len(value) > 500:
                    print(f"  {value[:500]}...")
                else:
                    print(f"  {value}")
    else:
        print("完整样本示例:")
        print("=" * 60)
        print("⚠️  数据集中没有可用的拆分或拆分为空")

    return dataset, corpus


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
    # 检查命令行参数
    dataset_key = "nfcorpus"  # 默认使用最小的数据集
    use_cache = True
    save_local = False
    force_download = False

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg in ["--list", "-l", "list"]:
                print_dataset_list()
                sys.exit(0)
            elif arg in ["--save-local", "-s"]:
                save_local = True
            elif arg in ["--force-download", "-f"]:
                force_download = True
                use_cache = False
            elif arg in ["--no-cache"]:
                use_cache = False
            elif not arg.startswith("--") and not arg.startswith("-"):
                # 如果不是选项，可能是数据集名称
                dataset_key = arg.lower()

    # 获取数据集信息
    if dataset_key in SMALL_DATASETS:
        # 使用白名单中的数据集信息
        dataset_info = SMALL_DATASETS[dataset_key]
        dataset_name = dataset_info["name"]
        print(f"\n📦 使用数据集: {dataset_key}")
        print(f"   {dataset_info['description']}")
        print(f"   大小: {dataset_info['size']}")
    else:
        # 使用用户指定的数据集名称（自动添加 mteb/ 前缀如果不存在）
        if dataset_key.startswith("mteb/"):
            dataset_name = dataset_key
        else:
            dataset_name = f"mteb/{dataset_key}"
        print(f"\n📦 使用数据集: {dataset_name}")
        print(f"   (用户指定的数据集，不在推荐列表中)")
    if save_local:
        print("   💾 将保存到本地以便下次快速加载")
    if force_download:
        print("   ⚠️  将强制重新下载")
    print()

    dataset, corpus = view_dataset_data(dataset_name, use_cache=use_cache, save_local=save_local)

    # 可选：进一步分析
    print("\n" + "=" * 60)
    print("数据统计:")
    print("=" * 60)

    # 显示所有拆分的统计信息
    for split_name in dataset.keys():
        split_data = dataset[split_name]
        if len(split_data) == 0:
            continue

        print(f"\n{split_name.upper()} 拆分统计:")
        print(f"  - 总样本数: {len(split_data)}")

        # 计算平均查询长度（如果存在 query 字段）
        if len(split_data) > 0:
            first_item = split_data[0]
            # 尝试找到包含查询文本的字段
            query_field = None
            for key in first_item.keys():
                if 'query' in key.lower() or 'text' in key.lower():
                    query_field = key
                    break

            if query_field:
                sample_size = min(1000, len(split_data))
                query_lengths = [len(str(item[query_field])) for item in split_data.select(range(sample_size))]
                avg_query_len = sum(query_lengths) / len(query_lengths) if query_lengths else 0
                print(f"  - 平均{query_field}长度 (前{sample_size}个样本): {avg_query_len:.1f} 字符")

