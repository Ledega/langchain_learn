import os  # 操作系统接口模块，本例仅用来从环境变量读取配置（避免硬编码密钥）
from openai import OpenAI  # OpenAI 客户端类；实例化后可直接发起聊天补全等请求

client = OpenAI(  # 创建客户端实例，后续所有 API 调用都通过它发出
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # 读取 DASHSCOPE_API_KEY 环境变量作为鉴权凭证
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 使用 DashScope 的 OpenAI 兼容模式地址
)  # 客户端初始化结束

completion = client.chat.completions.create(  # 发送聊天补全请求，返回响应对象（含模型输出、用量等）
    model="qwen-plus",  # 选择具体模型名，不同模型能力与成本不同
    messages=[  # 对话历史列表，顺序决定模型上下文
        {"role": "system", "content": "You are a helpful assistant."},  # system 角色用于约束助手行为
        {"role": "user", "content": "who are you?"}  # user 角色表示用户输入
    ]  # messages 列表结束
)  # 请求结束，completion 内含 choices、usage 等字段

print(completion.model_dump_json)  # 打印 model_dump_json 方法对象（需要调用它才会输出 JSON 字符串）