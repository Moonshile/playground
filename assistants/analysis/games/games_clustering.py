#!/usr/bin/env python3
"""
games_clustering.py

功能：
对游戏数据进行聚类分析，将相似的游戏归类到一起。
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import sys
import pandas as pd
import os


def main():
    if len(sys.argv) < 4:
        print("用法: python games_clustering.py <input_csv_file> <output_csv_file> <n_clusters> [encoding]")
        print("参数说明:")
        print("  input_csv_file: 输入的CSV文件路径（包含游戏名称、游戏规则等列）")
        print("  output_csv_file: 输出的CSV文件路径")
        print("  n_clusters: 聚类类别数量（整数）")
        print("  encoding: 可选，文件编码（默认: utf-8）")
        print("\n示例:")
        print("  python games_clustering.py input.csv output.csv 10")
        print("  python games_clustering.py input.csv output.csv 15 utf-8")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    try:
        n_clusters = int(sys.argv[3])
        if n_clusters <= 0:
            print("❌ 错误: 类别数量必须大于0")
            sys.exit(1)
    except ValueError:
        print("❌ 错误: 类别数量必须是整数")
        sys.exit(1)

    encoding = sys.argv[4] if len(sys.argv) > 4 else 'utf-8'

    try:
        # 检查输入文件是否存在
        if not os.path.exists(input_file):
            print(f"❌ 错误: 输入文件不存在: {input_file}")
            sys.exit(1)

        print(f"📖 正在读取输入文件: {input_file}")
        df = pd.read_csv(input_file, encoding=encoding)
        print(f"✅ 读取完成，共 {len(df)} 条记录")

        # 检查必要的列是否存在
        required_columns = ["游戏名称", "游戏规则"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"❌ 错误: 输入文件缺少必要的列: {', '.join(missing_columns)}")
            sys.exit(1)

        print(f"🔄 正在进行聚类分析（类别数: {n_clusters}）...")

        # 使用游戏规则和名称作为文本特征
        texts = df["游戏名称"].astype(str) + " " + df["游戏规则"].astype(str)

        # 提取文本特征
        vectorizer = TfidfVectorizer(max_features=500, stop_words=None)
        X = vectorizer.fit_transform(texts)

        # 使用 KMeans 聚类自动归类游戏
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        df["游戏类别ID"] = labels

        print(f"✅ 聚类完成，共生成 {len(set(labels))} 个类别")

        # 聚合同一类别的样本
        grouped_rows = []
        for label in sorted(df["游戏类别ID"].unique()):
            subset = df[df["游戏类别ID"] == label]
            category_name = subset["游戏名称"].mode()[0] if not subset["游戏名称"].mode().empty else f"类别{label}"
            rule_summary = subset["游戏规则"].iloc[0]

            # 用户列表示例（最多10个用户链接）
            user_links = []
            if "聊天记录" in subset.columns:
                user_links = subset["聊天记录"].tolist()
            # 限制最多10个用户链接
            max_user_links = 10
            user_links = user_links[:max_user_links]
            user_list_sample = "\n".join(user_links)
            # 如果原始链接数量超过限制，添加提示
            if "聊天记录" in subset.columns and len(subset["聊天记录"].tolist()) > max_user_links:
                user_list_sample += f"\n(共{len(subset)}个用户，仅显示前{max_user_links}个)"

            # 对话示例（不同用户之间用 --- 分隔）
            dialogs = []
            if "聊天内容示例" in subset.columns:
                dialogs = subset["聊天内容示例"].tolist()
            dialog_sample = "\n---\n".join(dialogs[:10])  # 最多10个样本

            grouped_rows.append({
                "游戏类别": category_name,
                "玩法规则": rule_summary,
                "用户列表示例": user_list_sample,
                "对话示例": dialog_sample,
                "样本数量": len(subset)
            })

        # 构造结果 DataFrame
        result_df = pd.DataFrame(grouped_rows)

        print(f"💾 正在保存结果到: {output_file}")
        result_df.to_csv(output_file, index=False, encoding=encoding)

        print(f"✅ 完成！")
        print(f"   输入记录数: {len(df)}")
        print(f"   聚类类别数: {len(result_df)}")
        print(f"   结果已保存到: {output_file}")

    except FileNotFoundError:
        print(f"❌ 错误: 文件不存在: {input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
