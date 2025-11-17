import streamlit as st
from openai import OpenAI

from rag.db_loader import load_all_dbs
from rag.retrievers import create_retrievers
from rag.context_chain import build_chain
from langchain_openai import ChatOpenAI

client = OpenAI()

st.set_page_config(page_title="프로모션 RAG 챗봇", layout="wide")
st.title("🎉 아티제 프로모션 추천 챗봇")


# RAG 시스템 준비
# @st.cache_resource
# def init_rag():
#     db_region, db_market, db_artisee, db_promo = load_all_dbs()


#     st.write("db_region type:", type(db_region))
#     st.write("db_market type:", type(db_market))
#     st.write("db_artisee type:", type(db_artisee))
#     st.write("db_promo type:", type(db_promo))

#     retr_region, retr_market, retr_artisee, retr_promo = create_retrievers(
#         db_region, db_market, db_artisee, db_promo
#     )
#     return db_region, db_market, retr_artisee, retr_promo

@st.cache_resource
def init_rag():
    db_region, db_market, db_artisee, db_promo = load_all_dbs()
    retr_region, retr_market, retr_artisee, retr_promo = create_retrievers(
        db_region, db_market, db_artisee, db_promo
    )
    
    return (
        db_region,
        db_market,
        db_artisee,
        db_promo,
        retr_region,
        retr_market,
        retr_artisee,
        retr_promo
    )

#db_region, db_market, retr_artisee, retr_promo = init_rag()

(
    db_region,
    db_market,
    db_artisee,
    db_promo,
    retr_region,
    retr_market,
    retr_artisee,
    retr_promo
) = init_rag()

# st.write(type(retr_artisee))
# st.write(type(retr_promo))

# LLM 준비
#llm = client.chat.completions.with_options(model="gpt-4o-mini")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

chain = build_chain(
    db_region=db_region,
    db_market=db_market,
    retriever_artisee=retr_artisee,
    retriever_promo=retr_promo,
    llm=llm
)
# Streamlit UI
if "history" not in st.session_state:
    st.session_state["history"] = ""

# query = st.text_input("질문을 입력하세요:")

# #st.write("Test search:", retr_artisee.get_relevant_documents("아티제 소개 알려줘"))

# if st.button("전송"):
#     result = chain.invoke({
#         "question": query,
#         "chat_history": st.session_state["history"]
#     })

#     st.session_state["history"] += f"\n사용자: {query}\nAI: {result}\n"

#     st.write(result)
import streamlit as st
from streamlit_chat import message

st.set_page_config(page_title="Artisee RAG Chatbot", page_icon="☕", layout="wide")

# --- UI HEADER ---
st.markdown(
    """
    <h2 style='text-align: center;'>☕ Artisee AI 프로모션 컨설턴트</h2>
    <p style='text-align: center; color: gray;'>상권·지역·브랜드 기반 RAG 챗봇</p>
    <hr>
    """,
    unsafe_allow_html=True
)

# --- 채팅 히스토리 관리 ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 이전 메시지 렌더링
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        message(msg["content"], is_user=True, key=msg["key"])
    else:
        message(msg["content"], is_user=False, key=msg["key"])

# --- 사용자 입력 영역 ---
user_input = st.chat_input("메시지를 입력하세요...")

if user_input:
    # ① 사용자 메시지 기록
    st.session_state["messages"].append({
        "role": "user",
        "content": user_input,
        "key": f"user_{len(st.session_state['messages'])}"
    })

    # ② RAG 체인 실행
    result = chain.invoke({
        "question": user_input,
        "chat_history": "\n".join(
            [f"{m['role']}: {m['content']}" for m in st.session_state["messages"]]
        )
    })

    # ③ AI 메시지 기록
    st.session_state["messages"].append({
        "role": "assistant",
        "content": result,
        "key": f"ai_{len(st.session_state['messages'])}"
    })

    # ④ 즉시 리렌더링
    st.rerun()
