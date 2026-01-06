# 如何进行向量数据库测试

## 数据描述

数据源位于 `.data/vectors` 目录下，分别是由 OpenAI、gemini文本模型、gemini多模态模型生成的向量。

## 向量数据库描述

## 任务描述

- 基于向量化评测生成的q-d向量
- 所有document向量全部入库后再开始评测
- 评测基准：基于暴力法算出来的top-N最近邻向量
- 评测指标
  - 检索能力：每个请求分别计算准确率和召回率，最后计算所有请求的平均值
  - 处理能力：记录每个请求的时间，并报告最终的时间分布（最大最小、平均、分位数）

## 向量库使用方式

### Milvus

- 连接 [https://docs.zilliz.com/docs/quick-start#set-up-connection]
  - CLUSTER_ENDPOINT 从环境变量 MILVUS_CLUSTER_ENDPOINT 中读取
  - TOKEN 从环境变量 MILUVS_TOKEN 中读取
- 创建 collection [https://docs.zilliz.com/docs/quick-start#create-collection]
  - 为每种模型生成的数据集各自分别创建一个collection
  - 字段列表
    - primary_key: document的sha2048哈希值，类型是 VARCHAR (1024)
    - vector: 向量，类型是 FLOAT_VECTOR，建索引
- 插入数据 [https://docs.zilliz.com/docs/quick-start#create-collection]
- 检索数据 [https://docs.zilliz.com/docs/quick-start#similarity-search]
