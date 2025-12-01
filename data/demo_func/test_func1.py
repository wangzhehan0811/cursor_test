"""
Function Call 调用流程演示

本文件演示了如何使用阿里云 DashScope API 进行 Function Calling（函数调用）的完整流程。

Function Call 调用路径：
1. 用户发送查询 -> run_conversation()
2. 第一次 API 调用 -> get_response() -> dashscope.Generation.call()
3. 模型返回 function_call 请求 -> 解析 function_call
4. 执行本地函数 -> get_current_weather()
5. 将函数结果添加到消息历史 -> messages.append(tool_info)
6. 第二次 API 调用 -> get_response() -> dashscope.Generation.call()
7. 模型返回最终答案 -> 返回给用户

详细流程说明：
- Step 1: 用户输入查询，调用 run_conversation()
- Step 2: 第一次调用 API，模型分析是否需要调用函数
- Step 3: 如果模型决定调用函数，返回 function_call 对象
- Step 4: 解析 function_call，提取函数名和参数
- Step 5: 执行对应的本地函数（如 get_current_weather）
- Step 6: 将函数执行结果添加到消息历史中
- Step 7: 第二次调用 API，模型基于函数结果生成最终回答
"""

import json
import os
import subprocess
import dashscope  # type: ignore


def load_bash_profile():
    """
    加载 ~/.bash_profile 文件中的环境变量
    
    通过执行 shell 命令来加载 .bash_profile，并更新当前进程的环境变量
    这样可以确保从 .bash_profile 中导出的环境变量（如 DASHSCOPE_API_KEY）可用
    """
    try:
        # 使用 subprocess 执行 shell 命令来加载 .bash_profile
        # 通过执行 'source ~/.bash_profile && env' 来获取所有环境变量
        result = subprocess.run(
            ['bash', '-c', 'source ~/.bash_profile && env'],
            capture_output=True,
            text=True,
            check=True
        )
        
        # 解析环境变量并更新到当前进程
        for line in result.stdout.splitlines():
            if '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value
        
        print("成功加载 ~/.bash_profile 中的环境变量")
    except subprocess.CalledProcessError as e:
        print(f"加载 ~/.bash_profile 失败: {e}")
    except FileNotFoundError:
        print("未找到 ~/.bash_profile 文件，跳过加载")
    except Exception as e:
        print(f"加载环境变量时出错: {e}")


# 在设置 API key 之前，先加载 .bash_profile 中的环境变量
load_bash_profile()

# 从环境变量中获取 DASHSCOPE_API_KEY
api_key = os.environ.get('DASHSCOPE_API_KEY')
print('api_key=', api_key)
dashscope.api_key = api_key

def get_current_weather(location, unit="摄氏度"):
    """
    获取指定地点的当前天气信息
    
    这是一个可被大模型调用的工具函数。在实际应用中，可以调用高德地图API等
    第三方服务获取实时天气数据。为了演示流程，这里使用固定的天气数据。
    
    调用路径：
    - 被 run_conversation() 中的 function call 流程调用
    - 参数来自模型解析的 function_call['arguments']
    
    参数:
        location (str): 城市名称，支持中文或英文（如：大连、Dalian、上海、Shanghai等）
        unit (str): 温度单位，默认为"摄氏度"
    
    返回:
        str: JSON格式的天气信息字符串，包含地点、温度、单位和预报
    """
    # 初始化默认温度值
    temperature = -1
    
    # 根据城市名称匹配对应的温度（演示用固定数据）
    if '大连' in location or 'Dalian' in location:
        temperature = 10
    if '上海' in location or 'Shanghai' in location:
        temperature = 36
    if '深圳' in location or 'Shenzhen' in location:
        temperature = 37
    
    # 构建天气信息字典
    weather_info = {
        "location": location,
        "temperature": temperature,
        "unit": unit,
        "forecast": ["晴天", "微风"],
    }
    
    # 返回 JSON 格式的字符串（模型需要字符串格式的函数返回值）
    return json.dumps(weather_info)

def get_response(messages):
    """
    调用 DashScope API 获取模型响应
    
    调用路径：
    - 被 run_conversation() 调用（可能调用两次）
    - 第一次：发送用户查询，模型决定是否需要调用函数
    - 第二次：发送函数执行结果，模型生成最终回答
    
    参数:
        messages (list): 消息历史列表，包含用户消息、助手消息和函数消息
    
    返回:
        response: DashScope API 响应对象，包含模型的输出
        None: 如果 API 调用失败则返回 None
    """
    try:
        # 调用 DashScope Generation API
        # functions 参数告诉模型有哪些可用的函数可以调用
        response = dashscope.Generation.call(
            model='qwen-max',           # 使用的模型名称
            messages=messages,           # 对话历史消息
            functions=functions,         # 可用的函数定义列表
            result_format='message'     # 返回格式为消息格式
        )
        return response
    except Exception as e:
        print(f"API调用出错: {str(e)}")
        return None

def run_conversation():
    """
    使用 Function Call 进行问答的主流程函数
    
    Function Call 完整调用路径：
    ┌─────────────────────────────────────────────────────────────┐
    │ Step 1: 用户输入查询                                         │
    │   query = "大连的天气怎样"                                    │
    │   messages = [{"role": "user", "content": query}]           │
    └─────────────────────────────────────────────────────────────┘
                          ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ Step 2: 第一次 API 调用                                       │
    │   get_response(messages)                                     │
    │   → dashscope.Generation.call()                              │
    │   → 模型分析查询，决定需要调用 get_current_weather 函数      │
    │   → 返回 function_call 对象                                  │
    └─────────────────────────────────────────────────────────────┘
                          ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ Step 3: 解析 Function Call                                   │
    │   message.function_call = {                                  │
    │     'name': 'get_current_weather',                          │
    │     'arguments': '{"location": "大连", "unit": "celsius"}'   │
    │   }                                                          │
    └─────────────────────────────────────────────────────────────┘
                          ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ Step 4: 执行本地函数                                          │
    │   get_current_weather(location="大连", unit="celsius")       │
    │   → 返回天气信息 JSON 字符串                                 │
    └─────────────────────────────────────────────────────────────┘
                          ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ Step 5: 将函数结果添加到消息历史                              │
    │   tool_info = {                                              │
    │     "role": "function",                                      │
    │     "name": "get_current_weather",                          │
    │     "content": "{\"temperature\": 10, ...}"                  │
    │   }                                                          │
    │   messages.append(tool_info)                                 │
    └─────────────────────────────────────────────────────────────┘
                          ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ Step 6: 第二次 API 调用                                       │
    │   get_response(messages)  # 包含函数执行结果                 │
    │   → dashscope.Generation.call()                              │
    │   → 模型基于函数返回的天气数据生成最终回答                    │
    └─────────────────────────────────────────────────────────────┘
                          ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ Step 7: 返回最终答案给用户                                    │
    │   return message  # 包含模型生成的最终回答                   │
    └─────────────────────────────────────────────────────────────┘
    
    返回:
        message: 模型的最终回答消息对象
        None: 如果执行失败则返回 None
    """
    # ========== Step 1: 初始化用户查询 ==========
    query = "大连的天气怎样"
    messages = [{"role": "user", "content": query}]
    
    # ========== Step 2: 第一次 API 调用 ==========
    # 调用模型，模型会分析用户查询，决定是否需要调用函数
    # 如果模型判断需要调用函数，会在响应中包含 function_call 对象
    response = get_response(messages)
    if not response or not response.output:
        print("获取响应失败")
        return None
        
    print('第一次响应 response=', response)
    
    # 提取模型返回的消息
    message = response.output.choices[0].message
    # 将模型消息添加到对话历史中
    messages.append(message)
    print('模型消息 message=', message)
    
    # ========== Step 3: 判断是否需要执行 Function Call ==========
    # 检查模型是否返回了 function_call 请求
    if hasattr(message, 'function_call') and message.function_call:
        # ========== Step 4: 解析 Function Call 信息 ==========
        function_call = message.function_call
        tool_name = function_call['name']  # 获取要调用的函数名
        print(f'模型请求调用函数: {tool_name}')
        
        # 解析函数参数（JSON 字符串格式）
        arguments = json.loads(function_call['arguments'])
        print('函数参数 arguments=', arguments)
        
        # ========== Step 5: 执行本地函数 ==========
        # 根据函数名和参数调用对应的本地函数
        # 这里调用 get_current_weather 函数
        tool_response = get_current_weather(
            location=arguments.get('location'),
            unit=arguments.get('unit'),
        )
        print(f'函数执行结果: {tool_response}')
        
        # ========== Step 6: 构建函数执行结果消息 ==========
        # 将函数执行结果按照 API 要求的格式添加到消息历史中
        # role 必须是 "function"，name 是函数名，content 是函数返回结果
        tool_info = {
            "role": "function",
            "name": tool_name,
            "content": tool_response
        }
        print('函数消息 tool_info=', tool_info)
        
        # 将函数执行结果添加到消息历史
        messages.append(tool_info)
        print('更新后的消息历史 messages=', messages)
        
        # ========== Step 7: 第二次 API 调用 ==========
        # 将包含函数执行结果的消息历史发送给模型
        # 模型会基于函数返回的数据生成最终的用户友好回答
        response = get_response(messages)
        if not response or not response.output:
            print("获取第二次响应失败")
            return None
            
        print('第二次响应 response=', response)
        
        # 提取最终答案
        message = response.output.choices[0].message
        return message
    
    # 如果模型没有请求调用函数，直接返回模型的消息
    return message

# ========== Function 定义：告诉模型有哪些可用的函数 ==========
# 这个列表定义了模型可以调用的函数及其参数格式
# 模型会根据用户查询和函数描述，决定是否需要调用这些函数
functions = [
    {
        'name': 'get_current_weather',  # 函数名称，必须与实际的函数名一致
        'description': '获取指定地点的当前天气信息。',  # 函数描述，模型用这个来判断是否需要调用
        'parameters': {
            'type': 'object',
            'properties': {
                'location': {
                    'type': 'string',
                    'description': '城市名称，例如：大连、上海、深圳、San Francisco, CA'
                },
                'unit': {
                    'type': 'string',
                    'enum': ['celsius', 'fahrenheit'],  # 温度单位枚举值
                    'description': '温度单位，celsius 表示摄氏度，fahrenheit 表示华氏度'
                }
            },
            'required': ['location']  # 必填参数列表
        }
    }
]

if __name__ == "__main__":
    """
    主程序入口
    
    执行流程：
    1. 调用 run_conversation() 启动 Function Call 流程
    2. 打印最终结果
    """
    result = run_conversation()
    if result:
        print("=" * 50)
        print("最终结果:", result)
        print("=" * 50)
    else:
        print("对话执行失败")

