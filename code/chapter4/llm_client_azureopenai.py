import os
from openai import AzureOpenAI
from dotenv import load_dotenv
from typing import List, Dict

# 加载 .env 文件中的环境变量；override=True 让 .env 优先于系统中已有的同名变量
load_dotenv(override=True)

class HelloAgentsLLM:
    """
    为本书 "Hello Agents" 定制的LLM客户端（Azure OpenAI 版本）。
    接口与 llm_client.py 中的 HelloAgentsLLM 保持一致，默认使用流式响应。
    """
    def __init__(self, model: str = None, apiKey: str = None, endpoint: str = None,
                 apiVersion: str = None, timeout: int = None):
        """
        初始化客户端。优先使用传入参数，如果未提供，则从环境变量加载。
        在 Azure 中，model 参数填写的是“部署名称 (deployment name)”。
        """
        self.model = model or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        apiKey = apiKey or os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        apiVersion = apiVersion or os.getenv("AZURE_OPENAI_API_VERSION")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))

        if not all([self.model, apiKey, endpoint, apiVersion]):
            raise ValueError("部署名、API密钥、Endpoint 和 API 版本必须被提供或在.env文件中定义。")

        self.client = AzureOpenAI(
            api_key=apiKey,
            azure_endpoint=endpoint,
            api_version=apiVersion,
            timeout=timeout,
        )

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """
        调用大语言模型进行思考，并返回其响应。
        """
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )

            # 处理流式响应
            print("✅ 大语言模型响应成功:")
            collected_content = []
            for chunk in response:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            print()  # 在流式输出结束后换行
            return "".join(collected_content)

        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return None

# --- 客户端使用示例 ---
if __name__ == '__main__':
    try:
        llmClient = HelloAgentsLLM()

        exampleMessages = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": "写一个快速排序算法"}
        ]

        print("--- 调用LLM ---")
        responseText = llmClient.think(exampleMessages)
        if responseText:
            print("\n\n--- 完整模型响应 ---")
            print(responseText)

    except ValueError as e:
        print(e)
