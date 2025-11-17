"""
使用 Transformers 库直接调用 Qwen3-Embedding 模型进行文本嵌入
================================================================

本程序展示了如何使用 Hugging Face Transformers 库直接加载和使用 Qwen3-Embedding 模型，
而不是通过 SentenceTransformer 封装。这种方式提供了更底层的控制能力。

核心流程：
1. 使用 last_token_pool 函数从模型的最后一个 token 提取嵌入向量
2. 为查询文本添加任务指令，以指导模型理解任务类型
3. 对查询和文档进行批量编码
4. 通过归一化和点积计算相似度分数

与 SentenceTransformer 版本的区别：
- 直接使用 AutoModel 和 AutoTokenizer，更接近模型底层
- 手动实现池化操作（last_token_pool）
- 需要手动归一化嵌入向量
- 可以更灵活地控制模型的输入和输出
"""

import torch
import torch.nn.functional as F

from torch import Tensor
from modelscope import AutoTokenizer, AutoModel


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    """
    从最后一个 token 的隐藏状态中提取嵌入向量（池化操作）
    
    这是 Qwen3-Embedding 模型推荐的池化方式，使用序列的最后一个有效 token 的
    隐藏状态作为整个序列的表示。这种方法能够捕获整个序列的上下文信息。
    
    参数：
        last_hidden_states: 模型输出的最后一个隐藏层状态，形状为 [batch_size, seq_len, hidden_dim]
        attention_mask: 注意力掩码，形状为 [batch_size, seq_len]，1 表示有效 token，0 表示填充
    
    返回：
        池化后的嵌入向量，形状为 [batch_size, hidden_dim]
    """
    # 检查是否使用左填充（left padding）
    # 如果所有序列的最后一个位置都是有效 token（attention_mask 全为 1），则使用左填充
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    
    if left_padding:
        # 左填充情况下，最后一个 token 就是序列的实际最后一个 token
        return last_hidden_states[:, -1]
    else:
        # 右填充情况下，需要根据 attention_mask 找到每个序列的实际最后一个有效 token
        # sequence_lengths 计算每个序列的实际长度（减 1 是因为索引从 0 开始）
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        # 为每个样本选择其实际最后一个 token 的隐藏状态
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    """
    构建带任务指令的查询文本
    
    Qwen3-Embedding 模型支持指令式查询，通过添加任务描述可以帮助模型
    更好地理解查询意图，从而生成更准确的嵌入向量。
    
    参数：
        task_description: 任务描述，说明查询的目的（如检索、分类等）
        query: 实际的查询文本
    
    返回：
        格式化后的查询文本，格式为 "Instruct: {task_description}\nQuery:{query}"
    """
    return f'Instruct: {task_description}\nQuery:{query}'

# 定义任务描述：说明这是一个网络搜索查询任务，需要检索相关段落来回答问题
# 每个查询都需要附带一个一句话的指令来描述任务类型
task = 'Given a web search query, retrieve relevant passages that answer the query'

# 构建查询列表：为每个查询添加任务指令
# 查询文本会被格式化为 "Instruct: ...\nQuery: ..." 的形式
queries = [
    get_detailed_instruct(task, 'What is the capital of China?'),
    get_detailed_instruct(task, 'Explain gravity')
]

# 构建文档列表：检索文档不需要添加指令，直接使用原始文本
documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
]

# 合并查询和文档，准备批量处理
input_texts = queries + documents

# 加载分词器：使用左填充（padding_side='left'）以便在序列末尾进行池化操作
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B', padding_side='left')

# 加载模型：使用 ModelScope 的 AutoModel 加载 Qwen3-Embedding-0.6B 模型
model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-0.6B')

# 可选：启用 flash_attention_2 以获得更好的加速效果和内存节省
# 注意：需要安装 flash-attn 库，并且需要 GPU 支持
# model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-0.6B', attn_implementation="flash_attention_2", torch_dtype=torch.float16).cuda()

# 设置最大序列长度为 8192
max_length = 8192

# 对输入文本进行分词和编码
# padding=True: 自动填充到批次中最长序列的长度
# truncation=True: 如果序列超过 max_length，则截断
# return_tensors="pt": 返回 PyTorch 张量
batch_dict = tokenizer(
    input_texts,
    padding=True,
    truncation=True,
    max_length=max_length,
    return_tensors="pt",
)

# 将输入数据移动到模型所在的设备（CPU 或 GPU）
batch_dict.to(model.device)

# 通过模型进行前向传播，获取隐藏状态
outputs = model(**batch_dict)

# 使用 last_token_pool 函数从最后一个 token 提取嵌入向量
# outputs.last_hidden_state 是模型最后一层的隐藏状态
embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

# 对嵌入向量进行 L2 归一化
# 归一化后的向量可以用于计算余弦相似度（点积 = 余弦相似度）
embeddings = F.normalize(embeddings, p=2, dim=1)

# 计算查询和文档之间的相似度分数
# embeddings[:2] 是前两个查询的嵌入向量
# embeddings[2:] 是后面文档的嵌入向量
# 使用矩阵乘法计算所有查询与所有文档的点积（即余弦相似度）
scores = (embeddings[:2] @ embeddings[2:].T) * 100

# 打印相似度分数矩阵
# 结果矩阵 scores[i][j] 表示第 i 个查询与第 j 个文档的相似度分数
# 分数范围通常在 [-1, 1] 之间，越接近 1 表示越相似
print(scores.tolist())
# 预期输出示例：[[0.7645568251609802, 0.14142508804798126], [0.13549736142158508, 0.5999549627304077]]
# 第一个查询（中国首都）与第一个文档（北京）相似度高（0.76），与第二个文档（重力）相似度低（0.14）
# 第二个查询（解释重力）与第二个文档（重力）相似度高（0.60），与第一个文档（北京）相似度低（0.14）