"""
使用 Vertex AI Gemini 多模态 Embeddings 生成向量的测试脚本。
参考官方文档：https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/multimodal-embeddings-api?hl=zh-cn

用法示例：
    python vector/vectorize/gemini_multimodal_test.py -i input.json -o output.json --project YOUR_PROJECT --location us-central1
    python vector/vectorize/gemini_multimodal_test.py -i input.json -o output.json -m multimodalembedding@001 -r 5 --restart --max-items 10

输入文件格式（JSON 列表，每条至少包含 text，image/video 可选）：
[
  {
    "id": "item-1",
    "text": "一段文本",
    "image": "path/to/img.png",   # 支持本地路径或 gs://
    "video": "path/to/video.mp4"  # 支持本地路径或 gs://
  }
]
"""

import argparse
import base64
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from google.oauth2 import service_account
import vertexai
from vertexai.vision_models import Image, MultiModalEmbeddingModel, Video, VideoSegmentConfig

# 默认模型与鉴权范围
DEFAULT_MODEL = "multimodalembedding@001"
DEFAULT_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
# 费用（每百万 token），可通过环境变量覆盖：GEMINI_MULTI_PRICE_PER_M
DEFAULT_PRICE_PER_MILLION = float(os.getenv("GEMINI_MULTI_PRICE_PER_M", "2.0"))


# ==================== 工具与配置 ====================

def atomic_write(path: str, data: Dict[str, Any]) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def load_data(input_file: str) -> List[Dict[str, Any]]:
    print(f"📖 正在加载数据: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"✅ 已加载 {len(data)} 条数据")
    return data


def load_credentials():
    """
    从环境变量获取服务账号凭证：
    - GOOGLE_APPLICATION_CREDENTIALS: JSON 文件路径
    - GOOGLE_SERVICE_ACCOUNT_JSON: JSON 字符串
    - GOOGLE_SERVICE_ACCOUNT_JSON_B64: Base64 编码的 JSON
    若均不存在则返回 None，交由默认凭证（需本地已 gcloud auth application-default login）。
    """
    path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    json_str = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    json_b64 = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON_B64")

    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(f"GOOGLE_APPLICATION_CREDENTIALS 文件不存在: {path}")
        return service_account.Credentials.from_service_account_file(path, scopes=DEFAULT_SCOPES)

    if json_str:
        info = json.loads(json_str)
        return service_account.Credentials.from_service_account_info(info, scopes=DEFAULT_SCOPES)

    if json_b64:
        decoded = base64.b64decode(json_b64).decode("utf-8")
        info = json.loads(decoded)
        return service_account.Credentials.from_service_account_info(info, scopes=DEFAULT_SCOPES)

    return None


def init_vertex(project: str, location: str, credentials=None) -> None:
    vertexai.init(project=project, location=location, credentials=credentials)


def load_media(path: Optional[str], is_video: bool = False):
    if not path:
        return None
    if path.startswith("gs://"):
        return Video.load_from_file(path) if is_video else Image.load_from_file(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    return Video.load_from_file(path) if is_video else Image.load_from_file(path)


# ==================== 向量化核心 ====================

def embed_item(
    model: MultiModalEmbeddingModel,
    item: Dict[str, Any],
    video_segment_config: Optional[VideoSegmentConfig] = None,
) -> Dict[str, Any]:
    # 兼容无 text 的数据，优先 text，其次 query/document
    text = item.get("text") or item.get("query") or item.get("document") or ""
    image_path = item.get("image")
    video_path = item.get("video")

    image_obj = load_media(image_path, is_video=False) if image_path else None
    video_obj = load_media(video_path, is_video=True) if video_path else None

    # 如果三者都为空，返回 None 由上层跳过
    if not text and not image_obj and not video_obj:
        return None

    # 粗略估算 token，仅对文本；图像/视频不计入（缺少官方计费口径）
    estimated_tokens = int(len(text) * 0.25) if text else 0

    api_start = time.time()
    embeddings = model.get_embeddings(
        image=image_obj,
        video=video_obj,
        video_segment_config=video_segment_config,
        contextual_text=text if text else None,
        dimension=1408,
        # 文档示例使用 text_embedding；这里仍提供 contextual_text 以生成 text embedding
    )
    api_time = time.time() - api_start

    result: Dict[str, Any] = {
        "id": item.get("id"),
        "text": text,
        "image": image_path,
        "video": video_path,
        "api_time_seconds": api_time,
        "estimated_tokens": estimated_tokens,
    }

    # 提取 embedding
    if hasattr(embeddings, "text_embedding"):
        result["text_embedding"] = embeddings.text_embedding
    if hasattr(embeddings, "image_embedding") and embeddings.image_embedding:
        result["image_embedding"] = embeddings.image_embedding
    if hasattr(embeddings, "video_embeddings") and embeddings.video_embeddings:
        result["video_embeddings"] = [
            {
                "embedding": ve.embedding,
                "start_offset_sec": ve.start_offset_sec,
                "end_offset_sec": ve.end_offset_sec,
            }
            for ve in embeddings.video_embeddings
        ]

    # 统计向量维度
    result["dimensions"] = {
        "text": len(result["text_embedding"]) if "text_embedding" in result else 0,
        "image": len(result["image_embedding"]) if "image_embedding" in result else 0,
        "video": (
            len(result["video_embeddings"][0]["embedding"])
            if result.get("video_embeddings")
            else 0
        ),
    }

    return result


# ==================== 检查点与报告 ====================

def load_checkpoint(output_file: str) -> Dict[str, Any]:
    ckpt_path = output_file.replace(".json", "_checkpoint.json")
    if not os.path.exists(ckpt_path):
        return {
            "processed": 0,
            "results": [],
            "performance": [],
            "cumulative_api_time": 0.0,
            "cumulative_stats": {
                "prompt_tokens": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
            },
        }
    try:
        with open(ckpt_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        backup = ckpt_path + ".corrupted"
        os.replace(ckpt_path, backup)
        print(f"⚠️ 检查点损坏，已备份到 {backup}，将从头开始")
        return {
            "processed": 0,
            "results": [],
            "performance": [],
            "cumulative_api_time": 0.0,
            "cumulative_stats": {
                "prompt_tokens": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
            },
        }


def save_checkpoint(output_file: str, checkpoint: Dict[str, Any]) -> None:
    ckpt_path = output_file.replace(".json", "_checkpoint.json")
    atomic_write(ckpt_path, checkpoint)


def print_report(perf_log: List[Dict[str, Any]], total: int, model: str) -> None:
    if not perf_log:
        return
    last = perf_log[-1]
    print(f"📊 已处理: {last['processed']}/{total}, "
          f"API耗时: {last['api_time']:.2f}s, "
          f"累计耗时: {last['cumulative_api_time']:.2f}s, "
          f"速度: {last['items_per_sec']:.2f} 项/秒, "
          f"Token: {last['total_tokens']:,}, 费用: ${last['total_cost']:.4f}, "
          f"模型: {model}")


# ==================== 主流程 ====================

def process(
    input_file: str,
    output_file: str,
    project: str,
    location: str,
    model: str = DEFAULT_MODEL,
    batch_size: int = 1,
    from_scratch: bool = False,
    report_interval: int = 10,
    max_items: Optional[int] = None,
):
    credentials = load_credentials()
    if credentials:
        print("🔑 使用服务账号凭证初始化 Vertex AI")
    else:
        print("ℹ️ 未提供服务账号凭证，将尝试默认凭证（需已配置 ADC）")

    init_vertex(project, location, credentials=credentials)
    model_name = model if model else DEFAULT_MODEL
    mm_model = MultiModalEmbeddingModel.from_pretrained(model_name)

    data = load_data(input_file)
    if max_items:
        data = data[:max_items]

    if from_scratch:
        checkpoint = {
            "processed": 0,
            "results": [],
            "performance": [],
            "cumulative_api_time": 0.0,
        }
        print("🚀 从头开始（忽略已有检查点）")
    else:
        checkpoint = load_checkpoint(output_file)

    processed = checkpoint.get("processed", 0)
    results: List[Dict[str, Any]] = checkpoint.get("results", [])
    performance: List[Dict[str, Any]] = checkpoint.get("performance", [])
    cumulative_api = checkpoint.get("cumulative_api_time", 0.0)
    cumulative_stats = checkpoint.get("cumulative_stats", {
        "prompt_tokens": 0,
        "total_tokens": 0,
        "total_cost": 0.0,
    })
    start_index = processed

    print(f"✅ 恢复进度: 已处理 {processed} 条，剩余 {len(data) - processed} 条")

    skipped = 0
    last_batch_was_reported = False

    for batch_start in range(start_index, len(data), batch_size):
        batch = data[batch_start: batch_start + batch_size]
        for offset, item in enumerate(batch):
            idx = batch_start + offset
            try:
                embedded = embed_item(mm_model, item, video_segment_config=VideoSegmentConfig(end_offset_sec=1))

                # 如果数据为空（无 text/image/video），跳过但不报错
                if embedded is None:
                    skipped += 1
                    continue

                results.append(embedded)
                cumulative_api += embedded["api_time_seconds"]
                processed = idx + 1

                # 更新累计 token / cost
                estimated_tokens = embedded.get("estimated_tokens", 0)
                cumulative_stats["prompt_tokens"] += estimated_tokens
                cumulative_stats["total_tokens"] += estimated_tokens
                cumulative_stats["total_cost"] = (cumulative_stats["total_tokens"] / 1_000_000) * DEFAULT_PRICE_PER_MILLION

                perf = {
                    "processed": processed,
                    "api_time": embedded["api_time_seconds"],
                    "cumulative_api_time": cumulative_api,
                    "items_per_sec": processed / cumulative_api if cumulative_api > 0 else 0,
                    "prompt_tokens": cumulative_stats["prompt_tokens"],
                    "total_tokens": cumulative_stats["total_tokens"],
                    "total_cost": cumulative_stats["total_cost"],
                }
                performance.append(perf)

                # 在打印性能报告时保存检查点（逻辑分离：先打印报告，再保存检查点）
                if processed % report_interval == 0:
                    print_report(performance, len(data), model_name)
                    save_checkpoint(output_file, {
                        "processed": processed,
                        "results": results,
                        "performance": performance,
                        "cumulative_api_time": cumulative_api,
                        "cumulative_stats": cumulative_stats,
                        "model": model_name,
                    })
                    last_batch_was_reported = True
                elif idx == len(data) - 1:
                    # 最后一条，即使不满足报告间隔也要标记
                    last_batch_was_reported = False
            except Exception as e:
                print(f"❌ 处理第 {idx + 1} 条失败: {e}")
                save_checkpoint(output_file, {
                    "processed": processed,
                    "results": results,
                    "performance": performance,
                    "cumulative_api_time": cumulative_api,
                    "model": model_name,
                    "error": str(e),
                })
                raise

    # 处理完成，如果最后一批没有报告，则输出最终报告
    if not last_batch_was_reported and processed > 0:
        print_report(performance, len(data), model_name)
        save_checkpoint(output_file, {
            "processed": processed,
            "results": results,
            "performance": performance,
            "cumulative_api_time": cumulative_api,
            "cumulative_stats": cumulative_stats,
            "model": model_name,
        })

    # 计算最终统计
    total_api_time = cumulative_api
    effective_items = processed - skipped if processed > skipped else 0
    items_per_sec = effective_items / total_api_time if total_api_time > 0 else 0
    avg_time_per_item = total_api_time / effective_items if effective_items > 0 else 0.0

    # 终态成本/Token
    total_tokens = cumulative_stats["total_tokens"]
    total_cost = cumulative_stats["total_cost"]
    avg_tokens_per_item = total_tokens / effective_items if effective_items > 0 else 0

    # 保存最终结果
    print("💾 正在保存结果...")
    final_output = {
        "metadata": {
            "input_file": input_file,
            "model": model_name,
            "project": project,
            "location": location,
            "total_items": len(data),
            "processed": processed,
            "skipped_empty": skipped,
            "batch_size": batch_size,
            "report_interval": report_interval,
        },
        "results": results,
        "performance_log": performance,
        "performance_summary": {
            "total_api_time_seconds": round(total_api_time, 2),
            "items_per_second": round(items_per_sec, 2),
            "avg_time_per_item_seconds": round(avg_time_per_item, 4),
            "token_usage": {
                "total_prompt_tokens": cumulative_stats["prompt_tokens"],
                "total_tokens": total_tokens,
                "avg_tokens_per_item": round(avg_tokens_per_item, 2),
            },
            "cost": {
                "total_cost_usd": round(total_cost, 4),
                "price_per_million_tokens": DEFAULT_PRICE_PER_MILLION,
            },
        },
    }
    atomic_write(output_file, final_output)
    print(f"✅ 结果已保存到: {output_file}")

    # 删除检查点文件（处理完成）
    checkpoint_file = output_file.replace('.json', '_checkpoint.json')
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"🗑️  已删除检查点文件")

    # 打印最终性能统计（与其他两个脚本格式一致）
    print("\n" + "=" * 60)
    print("最终性能统计（仅API调用时间）")
    print("=" * 60)
    print(f"总处理项数: {len(data)}")
    if skipped > 0:
        print(f"  有效处理: {effective_items} 条")
        print(f"  跳过空项: {skipped} 条")
    print(f"总API调用时间: {total_api_time:.2f}s")
    print(f"处理速度: {items_per_sec:.2f} 项/秒")
    if effective_items > 0:
        print(f"\n平均每条数据耗时:")
        print(f"  总耗时: {avg_time_per_item:.4f}s/条")
    print(f"\nToken消耗统计:")
    print(f"  总输入Token: {cumulative_stats['prompt_tokens']:,}")
    print(f"  总Token: {total_tokens:,}")
    print(f"  平均每项Token: {avg_tokens_per_item:.1f}")
    print(f"\n费用统计:")
    print(f"  模型: {model_name}")
    print(f"  定价: ${DEFAULT_PRICE_PER_MILLION:.4f} / 百万tokens")
    print(f"  总费用: ${total_cost:.4f}")
    # 显示向量维度（从第一个有效结果中获取）
    if results:
        first_result = next((r for r in results if r.get("text_embedding") or r.get("image_embedding")), None)
        if first_result:
            dims = first_result.get("dimensions", {})
            text_dim = dims.get("text", 0)
            image_dim = dims.get("image", 0)
            video_dim = dims.get("video", 0)
            if text_dim > 0:
                print(f"\n向量维度:")
                print(f"  文本向量: {text_dim}")
                if image_dim > 0:
                    print(f"  图像向量: {image_dim}")
                if video_dim > 0:
                    print(f"  视频向量: {video_dim}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="测试 Vertex AI Gemini 多模态 Embeddings（文本/图片/视频）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python vector/vectorize/gemini_multimodal_test.py -i input.json -o output.json --project YOUR_PROJECT --location us-central1
  python vector/vectorize/gemini_multimodal_test.py -i input.json -o output.json --max-items 20 -r 5
        """
    )
    parser.add_argument("-i", "--input", required=True, help="输入 JSON 文件路径")
    parser.add_argument("-o", "--output", required=True, help="输出 JSON 文件路径")
    parser.add_argument(
        "--project",
        required=False,
        help="GCP 项目 ID（优先命令行，其次环境变量 GOOGLE_CLOUD_PROJECT/PROJECT_ID/GCP_PROJECT）",
    )
    parser.add_argument(
        "--location",
        required=False,
        help="Vertex AI 区域（优先命令行，其次环境变量 GOOGLE_CLOUD_LOCATION/VERTEX_LOCATION，默认 us-central1）",
    )
    parser.add_argument(
        "-m", "--model",
        default=DEFAULT_MODEL,
        help=f"使用的模型（默认: {DEFAULT_MODEL}）"
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=1,
        help="批量大小（多模态接口按条处理，默认1）"
    )
    parser.add_argument(
        "--restart", "--from-scratch",
        dest="from_scratch",
        action="store_true",
        help="从头开始处理（忽略已有检查点），但运行中仍会保存检查点"
    )
    parser.add_argument("-r", "--report-interval", type=int, default=10, help="性能报告间隔（条）")
    parser.add_argument("--max-items", type=int, default=None, help="最大处理条数（用于抽样测试）")

    args = parser.parse_args()

    # 环境变量兜底
    env_project = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("PROJECT_ID") or os.getenv("GCP_PROJECT")
    env_location = os.getenv("GOOGLE_CLOUD_LOCATION") or os.getenv("VERTEX_LOCATION")

    project = args.project or env_project
    location = args.location or env_location or "us-central1"

    if not project:
        raise ValueError("请通过 --project 或环境变量 GOOGLE_CLOUD_PROJECT / PROJECT_ID / GCP_PROJECT 设置项目 ID")

    # 创建输出目录
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    process(
        input_file=args.input,
        output_file=args.output,
        project=project,
        location=location,
        model=args.model,
        batch_size=args.batch_size,
        from_scratch=args.from_scratch,
        report_interval=args.report_interval,
        max_items=args.max_items,
    )


if __name__ == "__main__":
    main()

