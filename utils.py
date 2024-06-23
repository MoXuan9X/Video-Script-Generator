from langchain.prompts import ChatPromptTemplate  # 导入 ChatPromptTemplate，用于创建对话模板
from langchain_openai import ChatOpenAI  # 导入 ChatOpenAI，用于与 OpenAI API 交互
from langchain_community.utilities import WikipediaAPIWrapper  # 导入 WikipediaAPIWrapper，用于获取维基百科内容
import os  # 导入 os 库，用于获取环境变量

api_key = os.getenv("OPENAI_API_KEY")  # 获取环境变量中的 API 密钥


def generate_script(subject, video_length, creativity, api_key):
    """
    生成视频脚本的函数
    :param subject: 视频主题
    :param video_length: 视频时长（分钟）
    :param creativity: 创造力参数（0.0 到 AI.0）
    :param api_key: OpenAI API 密钥
    :return: 维基百科搜索结果、视频标题、视频脚本
    """

    # 创建标题生成模板
    title_template = ChatPromptTemplate.from_messages(
        [
            ("human", "请为'{subject}'这个主题的视频想一个吸引人的标题")  # 定义生成标题的模板，使用用户输入的主题
        ]
    )

    # 创建脚本生成模板
    script_template = ChatPromptTemplate.from_messages(
        [
            ("human",
             """你是一位短视频频道的博主。根据以下标题和相关信息，为短视频频道写一个视频脚本。
             视频标题：{title}，视频时长：{duration}分钟，生成的脚本的长度尽量遵循视频时长的要求。
             要求开头抓住眼球，中间提供干货内容，结尾有惊喜，脚本格式也请按照【开头、中间，结尾】分隔。
             整体内容的表达方式要尽量轻松有趣，吸引年轻人。
             脚本内容可以结合以下维基百科搜索出的信息，但仅作为参考，只结合相关的即可，对不相关的进行忽略：
             ```{wikipedia_search}```""")  # 定义生成脚本的模板，包含标题、时长和维基百科搜索结果
        ]
    )

    # 初始化 ChatOpenAI 模型，使用指定的 API 密钥、API 基址和创造力参数
    model = ChatOpenAI(openai_api_key=api_key, openai_api_base="https://api.aigc369.com/v1", temperature=creativity)

    title_chain = title_template | model  # 将模板与模型组合，生成标题
    script_chain = script_template | model  # 将模板与模型组合，生成脚本

    title = title_chain.invoke({"subject": subject}).content  # 调用模型生成视频标题

    search = WikipediaAPIWrapper(lang="zh")  # 创建维基百科搜索工具，指定语言为中文
    search_result = search.run(subject)  # 搜索主题相关的维基百科内容

    # 调用模型生成视频脚本，结合生成的标题、视频时长和维基百科搜索结果
    script = script_chain.invoke({"title": title, "duration": video_length,
                                  "wikipedia_search": search_result}).content

    return search_result, title, script  # 返回搜索结果、标题和脚本

# print(generate_script("sora模型",AI,0.7, os.getenv("OPENAI_API_KEY")))  # 测试函数
