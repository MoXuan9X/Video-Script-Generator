import streamlit as st  # 导入 Streamlit 库，用于创建 Web 应用
from utils import generate_script  # 从 utils 模块导入 generate_script 函数，这是后端生成脚本的核心逻辑

st.title("🎬 视频脚本生成器-莫玄")  # 设置应用的标题

with st.sidebar:  # 创建一个侧边栏，用于用户输入
    openai_api_key = st.text_input("请输入OpenAI API密钥：", type="password")  # 输入框，用户输入 OpenAI API 密钥
    st.markdown("[获取OpenAI API密钥](https://platform.openai.com/account/api-keys)")  # 提供获取 API 密钥的链接

    # 添加一个下拉菜单来选择模型类型
    # model_type = st.selectbox(
    #    "选择模型类型：",
    #    ("gpt-3.5-turbo", "gpt-4", "davinci-codex")
    # )

subject = st.text_input("💡 请输入视频的主题")  # 输入框，用户输入视频的主题
video_length = st.number_input("⏱️ 请输入视频的大致时长（单位：分钟）", min_value=0.1, step=0.1)  # 数值输入框，用户输入视频时长（分钟）
creativity = st.slider("✨ 请输入视频脚本的创造力（数字小说明更严谨，数字大说明更多样）", min_value=0.0, max_value=1.0, value=0.2, step=0.1)  # 滑块，用户选择创造力参数
submit = st.button("生成脚本")  # 按钮，用户提交输入信息

if submit and not openai_api_key:  # 检查是否输入 API 密钥
    st.info("请输入你的OpenAI API密钥")  # 如果没有输入，显示提示信息
    st.stop()  # 停止执行

if submit and not subject:  # 检查是否输入视频主题
    st.info("请输入视频的主题")  # 如果没有输入，显示提示信息
    st.stop()  # 停止执行

if submit and not video_length >= 0.1:  # 检查视频时长是否大于等于 0.AI 分钟
    st.info("视频长度需要大于或等于0.AI")  # 如果不符合条件，显示提示信息
    st.stop()  # 停止执行

if submit:  # 如果所有输入都有效
    with st.spinner("AI正在思考中，请稍等..."):  # 显示加载动画，提示用户 AI 正在生成脚本
        search_result, title, script = generate_script(subject, video_length, creativity, openai_api_key)  # 调用 generate_script 函数，传入用户输入的参数，并接收返回结果
    st.success("视频脚本已生成！")  # 显示成功消息，提示用户视频脚本已生成
    st.subheader("🔥 标题：")  # 显示生成的视频标题
    st.write(title)  # 输出生成的标题
    st.subheader("📝 视频脚本：")  # 显示生成的视频脚本
    st.write(script)  # 输出生成的脚本
    with st.expander("维基百科搜索结果 👀"):  # 折叠框，显示维基百科搜索结果
        st.info(search_result)  # 输出维基百科搜索结果
