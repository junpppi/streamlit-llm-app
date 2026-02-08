import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


load_dotenv()


def get_llm_response(input_text: str, expert_choice: str) -> str:
	system_message = ""
	if expert_choice == "A":
		system_message = (
			"あなたは運動の専門家です。安全性を重視し、年齢・体力・目的に"
			"配慮した実践的なアドバイスを簡潔に日本語で提供してください。"
		)
	else:
		system_message = (
			"あなたは栄養の専門家です。栄養バランスと継続性を重視し、"
			"実践しやすい食事のアドバイスを簡潔に日本語で提供してください。"
		)

	prompt = ChatPromptTemplate.from_messages(
		[
			("system", system_message),
			("human", "{user_input}"),
		]
	)

	llm = ChatOpenAI(
		model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
		temperature=0.7,
	)
	chain = prompt | llm | StrOutputParser()
	return chain.invoke({"user_input": input_text})


st.set_page_config(page_title="専門家AIアシスタント", page_icon="🧠")
st.title("🧠 専門家AIアシスタント")

st.markdown(
	"""
このアプリは、入力したテキストをLLMに渡して専門家としての回答を得るための
シンプルなデモです。以下の手順で操作できます。

1. 専門家の種類（A: 運動 / B: 栄養）を選択します。
2. 入力フォームに相談内容を入力します。
3. 送信すると、選択した専門家としての回答が表示されます。
"""
)

expert_label = st.radio(
	"専門家の種類を選択してください",
	options=["A: 運動の専門家", "B: 栄養の専門家"],
	horizontal=True,
)
expert_choice = "A" if expert_label.startswith("A") else "B"

with st.form("input_form"):
	user_input = st.text_area("入力テキスト", placeholder="例: 週3回の運動習慣を作りたい")
	submitted = st.form_submit_button("送信")

if submitted:
	if not user_input.strip():
		st.warning("入力テキストを入力してください。")
	else:
		with st.spinner("LLMが回答を生成中..."):
			response = get_llm_response(user_input, expert_choice)
		st.subheader("回答")
		st.write(response)
