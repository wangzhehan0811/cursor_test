"""
SentenceTransformer 原理说明
============================

SentenceTransformer 是一个用于生成句子嵌入（sentence embeddings）的框架，其核心原理如下：

1. 基础架构：
   - 基于 Transformer 架构（如 BERT、RoBERTa 等预训练模型）
   - 使用预训练的语言模型作为编码器（encoder）
   - 通过池化层（pooling layer）将变长的句子转换为固定维度的向量表示

2. 工作原理：
   - 输入处理：将文本句子通过 tokenizer 转换为 token IDs
   - 编码过程：通过 Transformer 编码器获取每个 token 的上下文表示
   - 池化操作：使用平均池化（mean pooling）、最大池化（max pooling）或 CLS token
     等方式将 token 级别的表示聚合成句子级别的固定维度向量
   - 输出：生成一个高维向量（通常 384、512、768 等维度），该向量捕获了句子的语义信息

3. 训练方式：
   - 通常使用对比学习（contrastive learning）方法进行训练
   - 通过正样本对（相似句子）和负样本对（不相似句子）进行对比
   - 优化目标是使相似句子的嵌入向量在向量空间中距离更近，不相似句子的距离更远

4. 应用场景：
   - 语义搜索：通过计算查询和文档的嵌入向量相似度进行检索
   - 文本相似度计算：使用余弦相似度或点积来衡量两个文本的相似程度
   - 聚类和分类：将文本转换为向量后可用于机器学习任务
   - 信息检索：在向量数据库中进行相似度搜索

5. Qwen3-Embedding 模型特点：
   - 支持双向编码（query 和 document 可以使用不同的 prompt）
   - 通过 prompt_name 参数可以指定不同的编码模式
   - 最大序列长度可配置，本示例设置为 8192
"""

from sentence_transformers import SentenceTransformer

# 模型路径：Qwen3-Embedding-0.6B 是一个 0.6B 参数的嵌入模型
model_dir = "/Users/wangzhehan/.cache/modelscope/hub/models/Qwen/Qwen3-Embedding-0.6B"

# 加载模型，trust_remote_code=True 允许执行模型中的自定义代码
model = SentenceTransformer(model_dir, trust_remote_code=True)

# 设置模型的最大序列长度为 8192，以适应更长的文本输入
model.max_seq_length = 8192

# 定义查询文本列表（用于搜索的查询语句）
queries = [
    "how much protein should a female eat",
    "summit define",
]

# 定义文档文本列表（待检索的文档内容）
documents = [
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
]

# 对查询文本进行编码，使用 "query" prompt 模式生成查询向量
query_embeddings = model.encode(queries, prompt_name="query")

# 对文档文本进行编码，生成文档向量（默认使用 document prompt）
document_embeddings = model.encode(documents)

# 计算查询向量和文档向量的相似度分数
# 使用矩阵乘法（@）计算点积，然后乘以 100 进行缩放
# 结果矩阵 scores[i][j] 表示第 i 个查询与第 j 个文档的相似度分数
scores = (query_embeddings @ document_embeddings.T) * 100

# 打印相似度分数矩阵
# 分数越高表示查询和文档的语义相似度越高
print(scores.tolist())
# 预期输出示例：[[70.00668334960938, 8.184843063354492], [14.62419319152832, 77.71407318115234]]
# 第一个查询与第一个文档相似度高（70.0），与第二个文档相似度低（8.18）
# 第二个查询与第二个文档相似度高（77.71），与第一个文档相似度低（14.62）
