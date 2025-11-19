#!/usr/bin/env python
# coding: utf-8

from PyPDF2 import PdfReader

from typing import List, Tuple
import os
import pickle
import logging
import sys
import inspect
from langchain_openai import OpenAI
from langchain_community.callbacks.manager import get_openai_callback
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


class DetailedFormatter(logging.Formatter):
    """自定义日志格式化器，支持类名和方法名"""
    
    def format(self, record):
        # 尝试从调用栈中获取类名
        class_name = ''
        try:
            # 使用 inspect 模块查找调用者
            stack = inspect.stack()
            for frame_info in stack[1:]:  # 跳过当前帧
                frame = frame_info.frame
                # 检查 'self' 参数是否存在（表示是类方法）
                if 'self' in frame.f_locals:
                    self_obj = frame.f_locals['self']
                    class_name = self_obj.__class__.__name__
                    break
        except Exception:
            pass
        
        # 构建位置信息
        location_parts = []
        if class_name:
            location_parts.append(f"类:{class_name}")
        location_parts.append(f"方法:{record.funcName}()")
        location_parts.append(f"文件:{record.filename}:{record.lineno}")
        location_info = " | ".join(location_parts)
        
        # 格式化日志消息
        log_format = (
            f'%(asctime)s | %(levelname)-8s | {location_info} | %(message)s'
        )
        formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)


# 配置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 创建控制台处理器
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(DetailedFormatter())

# 添加处理器到日志记录器
if not logger.handlers:
    logger.addHandler(console_handler)

DASHSCOPE_API_KEY = 'sk-882e296067b744289acf27e6e20f3ec0'


def load_qa_chain(llm, chain_type="stuff"):
    """
    兼容旧版 load_qa_chain 的函数，使用新的 LangChain LCEL API
    
    参数:
        llm: 语言模型实例
        chain_type: 链类型（目前只支持 "stuff"）
    
    返回:
        QAChain 对象，兼容旧版 API
    """
    # 创建提示模板
    prompt = ChatPromptTemplate.from_template("""
    基于以下上下文信息回答问题。如果上下文中没有相关信息，请说明无法从提供的文档中找到答案。
    
    上下文：
    {context}
    
    问题：{question}
    
    请根据上下文信息回答问题：
    """)
    
    # 创建文档链
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # 创建一个包装类以兼容旧版 API
    class QAChain:
        def invoke(self, input=None, **kwargs):
            """
            执行问答链
            
            参数:
                input: 包含 "input_documents" 和 "question" 的字典（可选）
                **kwargs: 如果 input 未提供，可以直接传递 input_documents 和 question
            
            返回:
                包含 "output_text" 的字典
            """
            # 支持两种调用方式：input= 或直接传参
            if input is not None:
                input_data = input
            else:
                input_data = kwargs
            
            # 确保 input_documents 是 Document 对象列表
            docs = input_data["input_documents"]
            # 验证文档对象类型
            if docs and isinstance(docs[0], str):
                # 如果传入的是字符串列表，需要转换为 Document 对象
                from langchain_core.documents import Document
                docs = [Document(page_content=doc) if isinstance(doc, str) else doc for doc in docs]
            
            # 调用文档链，传递文档对象和问题
            # create_stuff_documents_chain 期望接收 {"context": docs, "question": question} 格式
            result = document_chain.invoke({
                "context": docs,  # 传递文档对象列表，而不是文本字符串
                "question": input_data["question"]
            })
            
            return {"output_text": result}
    
    return QAChain()


def extract_text_with_page_numbers(pdf) -> Tuple[str, List[int]]:
    """
    从PDF中提取文本并记录每行文本对应的页码
    
    参数:
        pdf: PDF文件对象
    
    返回:
        text: 提取的文本内容
        page_numbers: 每行文本对应的页码列表
    """
    text = ""
    page_numbers = []

    for page_number, page in enumerate(pdf.pages, start=1):
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text
            page_numbers.extend([page_number] * len(extracted_text.split("\n")))
        else:
            logger.warning(f"第 {page_number} 页未找到文本。")

    return text, page_numbers


def process_text_with_splitter(text: str, page_numbers: List[int], save_path: str = None) -> FAISS:
    """
    处理文本并创建向量存储
    
    参数:
        text: 提取的文本内容
        page_numbers: 每行文本对应的页码列表
        save_path: 可选，保存向量数据库的路径
    
    返回:
        knowledgeBase: 基于FAISS的向量存储对象
    """
    # 创建文本分割器，用于将长文本分割成小块
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " ", ""],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    # 分割文本
    chunks = text_splitter.split_text(text)
    logger.info(f"文本被分割成 {len(chunks)} 个块。")
        
    # 创建嵌入模型
    embeddings = HuggingFaceEmbeddings(
        model_name="/Users/wangzhehan/.cache/modelscope/hub/models/Qwen/Qwen3-Embedding-0.6B",
        model_kwargs={'device': 'cpu'}
    )
    
    logger.info("创建嵌入模型：" + embeddings.model_name)

    
    # 从文本块创建知识库
    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    logger.info("已从文本块创建知识库。")
    
    # 存储每个文本块对应的页码信息
    page_info = {chunk: page_numbers[i] for i, chunk in enumerate(chunks)}
    logger.info(f"页码信息: {page_info}")

    knowledgeBase.page_info = page_info
    
    # 如果提供了保存路径，则保存向量数据库和页码信息
    if save_path:
        # 确保目录存在
        os.makedirs(save_path, exist_ok=True)
        
        # 保存FAISS向量数据库
        knowledgeBase.save_local(save_path)
        logger.info(f"向量数据库已保存到: {save_path}")
        
        # 保存页码信息到同一目录
        with open(os.path.join(save_path, "page_info.pkl"), "wb") as f:
            logger.info(f)
            pickle.dump(page_info, f)
        logger.info(f"页码信息已保存到: {os.path.join(save_path, 'page_info.pkl')}")

    return knowledgeBase



# 读取PDF文件
pdf_reader = PdfReader('./langchain_test1/浦发上海浦东发展银行西安分行个金客户经理考核办法.pdf')
# 提取文本和页码信息
text, page_numbers = extract_text_with_page_numbers(pdf_reader)
logger.debug(f"页码信息: {page_numbers}")
logger.debug(f"提取的文本内容（前200字符）: {text[:200]}...")

logger.info(f"提取的文本长度: {len(text)} 个字符。")
    
# 处理文本并创建知识库，同时保存到磁盘
save_dir = "./vector_db"
knowledgeBase = process_text_with_splitter(text, page_numbers, save_path=save_dir)


#------------------------------------------------------------------------------------------------
# 第二部分：加载本地大语言模型并执行问答任务
#------------------------------------------------------------------------------------------------


# ========== 加载本地大语言模型 ==========
# 使用本地部署的 Qwen3-0.6B 模型进行文本生成和问答任务
# 模型路径指向 ModelScope 缓存目录中的模型文件
model_path = "/Users/wangzhehan/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"
logger.info(f"正在加载本地模型: {model_path}")

try:
    # 加载分词器（Tokenizer）
    # 分词器负责将文本转换为模型可以理解的 token 序列，以及将生成的 token 转换回文本
    # trust_remote_code=True 允许从远程加载自定义代码（如模型特定的处理逻辑）
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 加载预训练的语言模型
    # AutoModelForCausalLM 是用于因果语言建模（文本生成）的模型类
    # 注意：在 macOS 上，MPS（Metal Performance Shaders）可能存在内存限制问题
    # 因此强制使用 CPU 以避免内存不足错误
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,  # 信任远程代码，允许执行模型仓库中的自定义代码
        # 使用 float32 数据类型（CPU 兼容性更好，避免 MPS 内存问题）
        torch_dtype=torch.float32,
        # 强制使用 CPU，避免 macOS MPS 内存不足问题
        device_map=None,
    )
    # 显式将模型移动到 CPU
    model = model.to("cpu")
    
    # 创建文本生成 pipeline
    # pipeline 封装了模型推理的完整流程，包括预处理、模型推理和后处理
    # device=0 表示使用 GPU，device=-1 表示使用 CPU
    pipe = pipeline(
        "text-generation",  # 任务类型：文本生成
        model=model,  # 使用的模型
        tokenizer=tokenizer,  # 使用的分词器
        device=-1,  # 强制使用 CPU（-1 表示 CPU，避免 MPS 内存问题）
        max_new_tokens=512,  # 生成的最大 token 数量，控制回答长度
        temperature=0.7,  # 温度参数：控制生成的随机性，值越大越随机，值越小越确定
        do_sample=True,  # 启用采样模式，允许生成多样化的回答
        pad_token_id=tokenizer.eos_token_id,  # 设置填充 token 为结束 token，用于批处理
    )
    
    # 将 HuggingFace pipeline 包装为 LangChain 的 LLM 接口
    # 这样可以使用 LangChain 的链式调用和工具集成功能
    llm = HuggingFacePipeline(pipeline=pipe)
    logger.info("本地模型加载完成")
    
except Exception as e:
    # 如果模型加载失败，记录错误并抛出异常
    logger.error(f"加载本地模型失败: {e}")
    raise

# ========== 执行问答任务 ==========
# 设置要查询的问题
# 注意：这里有两个 query 赋值，第二个会覆盖第一个，实际使用的是第二个问题
query = "客户经理被投诉了，投诉一次扣多少分"
#query = "客户经理每年评聘申报时间是怎样的？"

if query:
    # 步骤1：向量相似度搜索
    # 在向量数据库中搜索与查询问题最相关的文档块
    # k=5 表示返回最相似的 5 个文档块（根据实际文档块数量调整）
    # 注意：如果向量数据库中的文档数量少于 k，会自动返回所有可用文档
    docs = knowledgeBase.similarity_search(query, k=5)
    logger.info(f"检索到 {len(docs)} 个相关文档块")
    
    # 验证返回的文档对象类型（调试用）
    if docs:
        logger.info(f"第一个文档对象类型: {type(docs[0])}, 是否有 page_content: {hasattr(docs[0], 'page_content')}")

    # 步骤2：创建问答链
    # load_qa_chain 创建一个问答链，将检索到的文档和问题组合后发送给 LLM
    # chain_type="stuff" 表示将所有文档内容直接拼接后一起发送给模型（适合文档较少的情况）
    chain = load_qa_chain(llm, chain_type="stuff")

    # 步骤3：准备输入数据
    # 将检索到的文档和问题组织成字典格式，作为问答链的输入
    input_data = {"input_documents": docs, "question": query}

    # 步骤4：执行问答链
    # 模型会根据检索到的文档内容回答问题
    # 本地模型无需成本跟踪（与云端 API 不同）
    logger.info("开始执行问答链...",f"input_data: {input_data}")
    response = chain.invoke(input=input_data)
    logger.info("问答链执行完成")
    
    # 步骤5：输出回答结果
    # 打印模型生成的回答文本
    print(response["output_text"])
    print("来源:")

    # 步骤6：显示答案来源信息
    # 记录已显示的页码，避免重复显示
    unique_pages = set()

    # 遍历所有检索到的文档块，提取并显示它们的来源页码
    # 这有助于用户了解答案来自文档的哪些页面
    for doc in docs:
        # 获取文档块的文本内容
        text_content = getattr(doc, "page_content", "")
        # 从知识库的页码信息字典中查找该文档块对应的页码
        # 如果找不到，则显示"未知"
        source_page = knowledgeBase.page_info.get(
            text_content.strip(), "未知"
        )

        # 只显示未出现过的页码，避免重复输出
        if source_page not in unique_pages:
            unique_pages.add(source_page)
            print(f"文本块页码: {source_page}")




