import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate



from zhipuai import ZhipuAI



# ####
# # 请确保填写你的API Key
# api_key = "8b35bd0e8485274929df27077dc5ce88.5L31oDzUClISrU0g"
#
# client = ZhipuAI(
#     api_key=api_key,
#     base_url='https://open.bigmodel.cn/api/paas/v4/'  # 确保URL正确
# )
#
# api_key = "sk-MRUzE4gHSBZRJ8bIsVWfmnjv0nl8aR5EXTQayhCxKyPfAY1I"
# # 月之暗面科技有限公司的API端点
# api_endpoint = "https://api.moonshot.cn/v1/chat/completions"
# # 请求头部，包含API密钥
# headers = {
#     "Authorization": f"Bearer {api_key}",
#     "Content-Type": "application/json"
# }
# ####两个模型接口

###质谱模型请求格式
# try:
#     completion = client.chat.completions.create(
#         model="glm-4-flash",  # 请填写你要调用的模型名称
#         messages=[
#             {"role": "system", "content": "你是一个乐于回答各种问题的小助手，你的任务是提供专业、准确、有洞察力的建议。"},
#             {"role": "user",
#              "content": "我对太阳系的行星非常感兴趣，尤其是土星。请提供关于土星的基本信息，包括它的大小、组成、环系统以及任何独特的天文现象。"},
#         ],
#         max_tokens=100,
#         temperature=0.5
#     )

###月之暗面模型请求格式
# # 请求体，包含模型名称和消息内容
# data = {
#     "model": "moonshot-v1-8k",  # 使用的模型
#     "messages": [
#         {"role": "system", "content": "你是一个乐于回答各种问题的小助手，你的任务是提供专业、准确、有洞察力的建议。"},
#         {"role": "user", "content": "请提供关于土星的基本信息，包括它的大小、组成、环系统以及任何独特的天文现象。"}
#     ],
#     "max_tokens": 100,  # 最大生成的token数
#     "temperature": 0.5  # 控制生成文本的随机性
# }
#
# try:
#     # 发送POST请求
#     response = requests.post(api_endpoint, headers=headers, json=data)
#
#     # 检查响应状态码
#     if response.status_code == 200:
#         # 解析响应内容
#         response_data = response.json()
#         # 打印模型生成的文本
#         print(response_data['choices'][0]['message']['content'])
#     else:
#         print(f"Error: {response.status_code}")
#         print(response.text)
#
# except Exception as e:
#     print(f"An error occurred: {e}")





load_dotenv()

prompt_template = ChatPromptTemplate.from_messages(
    [
        ('system', "当用户提出一个问题时，你需要根据问题的关键词来判断是否属于隐私问题，如果是，则输出走端侧，反之，走云侧，你只需要输出端侧或者云侧。"),
        ('human', "{question}")
    ]
)

model = ChatOpenAI(
    model = 'glm-4',
    openai_api_base = "https://open.bigmodel.cn/api/paas/v4/",
    max_tokens = 2,
    temperature = 0.8
)

chain = prompt_template | model 
while True:
    question = input("你想问什么？")
    answer = chain.invoke(input = {'question': question})
    print(answer.content)