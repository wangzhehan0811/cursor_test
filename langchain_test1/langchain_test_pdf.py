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
            
            # 合并所有文档内容
            context = "\n\n".join([doc.page_content for doc in input_data["input_documents"]])
            
            # 调用文档链
            result = document_chain.invoke({
                "context": context,
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


# 加载本地模型
model_path = "/Users/wangzhehan/.cache/modelscope/hub/models/Qwen/Qwen3-4B"
logger.info(f"正在加载本地模型: {model_path}")

try:
    # 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    # 创建 pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    # 创建 LangChain LLM 包装器
    llm = HuggingFacePipeline(pipeline=pipe)
    logger.info("本地模型加载完成")
    
except Exception as e:
    logger.error(f"加载本地模型失败: {e}")
    raise

# 设置查询问题
query = "客户经理被投诉了，投诉一次扣多少分"
query = "客户经理每年评聘申报时间是怎样的？"
if query:
    # 执行相似度搜索，找到与查询相关的文档
    docs = knowledgeBase.similarity_search(query, k=10)
    logger.info(f"检索到 {len(docs)} 个相关文档块")

    # 加载问答链
    chain = load_qa_chain(llm, chain_type="stuff")

    # 准备输入数据
    input_data = {"input_documents": docs, "question": query}

    # 执行问答链（本地模型无需成本跟踪）
    logger.info("开始执行问答链...")
    response = chain.invoke(input=input_data)
    logger.info("问答链执行完成")
    print(response["output_text"])
    print("来源:")

    # 记录唯一的页码
    unique_pages = set()

    # 显示每个文档块的来源页码
    for doc in docs:
        text_content = getattr(doc, "page_content", "")
        source_page = knowledgeBase.page_info.get(
            text_content.strip(), "未知"
        )

        if source_page not in unique_pages:
            unique_pages.add(source_page)
            print(f"文本块页码: {source_page}")




