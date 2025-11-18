import streamlit as st
from openai import OpenAI

from rag.db_loader import load_all_dbs
from rag.retrievers import create_retrievers
#from rag.context_chain import build_chain
from langchain_openai import ChatOpenAI
from rag.graph_chain import build_rag_graph

client = OpenAI()

st.set_page_config(page_title="프로모션 RAG 챗봇", layout="wide")


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

# 2) LangGraph 기반 RAG 앱 생성
chain = build_rag_graph(
    db_region=db_region,
    db_market=db_market,
    retriever_artisee=retr_artisee,
    retriever_promo=retr_promo,
)

# # 3) 질문 들어오면:
# result = rag_app.invoke({
#     "question": user_input,
#     "chat_history": st.session_state.get("history", "")
# })

# answer = result["answer"]


# chain = build_chain(
#     db_region=db_region,
#     db_market=db_market,
#     retriever_artisee=retr_artisee,
#     retriever_promo=retr_promo,
#     llm=llm
# )
# Streamlit UI
if "history" not in st.session_state:
    st.session_state["history"] = ""


import streamlit as st

# =========================================
# STYLE: 아티제 톤 + 말풍선 + 아이콘 위치 조정
# =========================================
st.markdown("""
<style>
/* 전체 배경 */
body, .stApp {
    background-color: #f7f3ee !important;
}

/* 사이드바 전체 폭 조절 */
[data-testid="stSidebar"] {
    width: 240px !important;         /* 원하는 너비 */
    min-width: 240px !important;
    max-width: 240px !important;
}

/* ===== Left Sidebar ===== */
.sidebar-title {
    font-size: 20px;
    font-weight: bold;
    color: #6b4e47;
    margin-bottom: 10px;
}
.chat-history-item {
    padding: 8px 12px;
    border-radius: 8px;
    margin-bottom: 6px;
    cursor: pointer;
    background-color: #ece2da;
}
.chat-history-item:hover {
    background-color: #e2d6cd;
}



/* ===== Chat Area ===== */
.chat-bubble {
    max-width: 70%;
    padding: 12px 16px;
    border-radius: 14px;
    margin-bottom: 12px;
    line-height: 1.5;
    font-size: 16px;
}

/* 사용자 말풍선 (오른쪽) */
.user-row {
    display: flex;
    justify-content: flex-end;
    align-items: flex-start;
    gap: 8px;
}
.user-bubble {
    background-color: #e6d4c3;
    color: #3c2f2f;
    border: 1px solid #d6c3b3;
}
.user-icon {
    font-size: 28px;
    margin-top: 4px;
}

/* AI 말풍선 (왼쪽) */
.ai-row {
    display: flex;
    justify-content: flex-start;
    align-items: flex-start;
    gap: 8px;
}
.ai-bubble {
    background-color: #7b5e57;
    color: white;
    border: 1px solid #6b4e47;
}
.ai-icon {
    font-size: 28px;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)


# =========================================
# LEFT SIDEBAR: 대화 목록
# =========================================
from pathlib import Path
import base64

# 로고 경로
LOGO_PATH = Path(__file__).resolve().parent / "아티제로고.png"

# 사이드바 로고
with st.sidebar:
    st.markdown(
        f"""
        <div style='text-align:center; padding-top:10px; padding-bottom:5px;'>
            <img src='data:image/png;base64,{base64.b64encode(open(LOGO_PATH,"rb").read()).decode()}' 
                 style='width:50%; border-radius:2px;'/>
        </div>
        """,
        unsafe_allow_html=True
    )
st.sidebar.markdown("<hr>", unsafe_allow_html=True)

st.sidebar.markdown("<div class='sidebar-title'>💬 대화 목록</div>", unsafe_allow_html=True)

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}  # 여러 대화 저장
if "current_session" not in st.session_state:
    st.session_state.current_session = "default"

# 새 대화 버튼
if st.sidebar.button("➕ 새 대화"):
    new_id = f"chat_{len(st.session_state.chat_sessions)+1}"
    st.session_state.chat_sessions[new_id] = []
    st.session_state.current_session = new_id
    st.rerun()


# 기존 대화 목록 버튼 (제목 = 첫 사용자 메시지 앞 10글자)
for session_id, msgs in st.session_state.chat_sessions.items():

    # 제목 추출
    if len(msgs) > 0:
        # 첫 user 메시지를 찾아 제목 생성
        first_user_msg = next(
            (m["content"] for m in msgs if m["role"] == "user"),
            "새 대화"
        )
        title = first_user_msg[:10]  # 앞 10글자
    else:
        title = "새 대화"

    # 버튼 만들기
    if st.sidebar.button(f"💭 {title}"):
        st.session_state.current_session = session_id
        st.rerun()
st.sidebar.markdown("<hr>", unsafe_allow_html=True)

st.sidebar.markdown("### 모드 선택")

# debug_mode = st.sidebar.radio(
#     "출력 모드",
#     ["💬 일반 모드", "🧪 평가 모드"],
#     index=0
# )

debug_mode = st.sidebar.radio(
    "출력 모드",
    [
        "answer",
        "debug",
    ],
    format_func=lambda x: "💬 일반 모드" if x=="answer" else "🧪 디버그 모드"
)
st.sidebar.markdown("<hr>", unsafe_allow_html=True)

#st.sidebar.markdown("<br><br>", unsafe_allow_html=True)




# =========================================
# MAIN CHAT WINDOW
# =========================================

st.markdown(
    """
    <h2 style='text-align:center; color:#6b4e47;'>☕ Artisee AI 프로모션 컨설턴트</h2>
    <p style='text-align:center; color:#8a6f63;'>상권·지역·브랜드 기반 카페 RAG 시스템</p>
    <hr>
    """,
    unsafe_allow_html=True
)


if "mode" not in st.session_state:
    st.session_state["mode"] = "chat"

#st.sidebar.markdown("### 기능 메뉴")
if st.sidebar.button("📍 전국 아티제 매장 분포"):
    st.session_state["mode"] = "store_map"
    st.rerun()


if st.session_state["mode"] == "store_map":
    # 지도 or 분포도 렌더링
    st.subheader("📍 전국 아티제 매장 분포")

    st.image("./지역별 상권 개수.png")
    st.markdown("---")

    # 🔥 메인 본문 하단에 버튼 생성
    if st.button("💬 챗봇으로 돌아가기"):
        st.session_state["mode"] = "chat"
        st.rerun()

    st.stop()   # 챗봇 코드 실행을 완전히 막음

# ===============================
# MAIN CHAT WINDOW - 챗봇 모드
DEFAULT_GREETING = """
안녕하세요! 아티제 AI 프로모션 컨설턴트입니다. \n
마케팅 및 프로모션 전략에 대해 무엇이든 물어보세요.😀
"""

# 현재 대화 불러오기
session = st.session_state.current_session
if session not in st.session_state.chat_sessions:
    st.session_state.chat_sessions[session] = []

messages = st.session_state.chat_sessions[session]

# === AI가 먼저 인사하기 ===
if len(messages) == 0:
    messages.append({"role": "assistant", "content": DEFAULT_GREETING})

# # 메시지 렌더링 함수
# def render_message(role, content):
#     if role == "assistant":
#         st.markdown(
#             f"""
#             <div class="ai-row">
#                 <div class="ai-icon">🤖</div>
#                 <div class="chat-bubble ai-bubble">{content}</div>
#             </div>
#             """,
#             unsafe_allow_html=True
#         )
#     else:
#         st.markdown(
#             f"""
#             <div class="user-row">
#                 <div class="chat-bubble user-bubble">{content}</div>
#                 <div class="user-icon">🧑</div>
#             </div>
#             """,
#             unsafe_allow_html=True
#         )
from markdown2 import markdown

def render_message(role, content):
    if role == "assistant":
        # Markdown → HTML 변환
        html_content = markdown(content)

        st.markdown(
            f"""
            <div class="ai-row">
                <div class="ai-icon">🤖</div>
                <div class="chat-bubble ai-bubble">
                    {html_content}
            """,
            unsafe_allow_html=True
        )

    else:
        # user message 그대로
        st.markdown(
            f"""
            <div class="user-row">
                <div class="chat-bubble user-bubble">{content}</div>
                <div class="user-icon">👤</div>
            </div>
            """,
            unsafe_allow_html=True
        )



# 기존 메시지 표시
for msg in messages:
    render_message(msg["role"], msg["content"])


# ===============================
# 입력창
# ===============================
# user_input = st.chat_input("메시지를 입력하세요...")

# if user_input:
#     # 사용자 메시지 저장
#     messages.append({"role": "user", "content": user_input})

#     # RAG 실행
#     ai_answer = chain.invoke({
#         "question": user_input,
#         "chat_history": "\n".join([f"{m['role']}: {m['content']}" for m in messages])
#     })

#     # AI 응답 저장
#     messages.append({"role": "assistant", "content": ai_answer})

#     # 저장 후 rerun
#     st.rerun()

# ===============================
# 입력창
# ===============================
user_input = st.chat_input("메시지를 입력하세요...")

if user_input:
    # 사용자 메시지 저장
    messages.append({"role": "user", "content": user_input})

    # 먼저 화면에 사용자 메시지를 표시한 상태로 rerun
    st.rerun()


# ===============================
# "답변 생성 중…" 처리
# ===============================

# 가장 최근 메시지가 user면 = AI 답변이 필요한 상태
# --- AI 답변 대기 상태 확인 ---
if len(messages) > 0 and messages[-1]["role"] == "user":
    
    # placeholder에 로딩 메시지 표시
    loading = st.empty()
    loading.markdown(
        """
        <div class="ai-row">
            <div class="ai-icon">🤖</div>
            <div class="chat-bubble ai-bubble">답변을 생성하고 있어요...</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 실제 RAG 실행
    rag_result = chain.invoke({
        "question": messages[-1]["content"],
        "chat_history": "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    })
    # -------------------------------
    # 🔧 출력 모드 선택 분기
    # -------------------------------


    # 🎯 디버그 모드
    if debug_mode == "debug":
        st.subheader("🧪 LangGraph 전체 State")
        st.json(rag_result)
        st.stop()  # 🔥 매우 중요! rerun 방지 → JSON 유지

    # 🎯 일반 모드
    ai_answer = "\n" + rag_result.get("answer", "")


    # 로딩 문구 제거
    loading.empty()


    # 완료 후 메시지 저장
    messages.append({"role": "assistant", "content": ai_answer})

    st.rerun()

# import streamlit as st
# import os

# OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
