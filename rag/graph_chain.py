# rag/graph_chain.py (예시)

from typing import TypedDict, List, Optional, Dict
import json

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

from rag.retrievers import StrictRetriever   # 네가 만든 그거
from operator import itemgetter


# -----------------------
# 1. 그래프 상태 정의
# -----------------------
class RAGState(TypedDict, total=False):
    question: str
    chat_history: str

    # router 결과
    need_rag: bool
    rag_sources: List[str]  # 예: ["region_market", "artisee", "promo"]

    # RAG 결과 컨텍스트
    local: str       # 지역 + 상권 인사이트 (StrictRetriever)
    artisee: str     # 아티제 브랜드 정보
    promo: str       # 프로모션 전략 지식

    region: Optional[str]
    market_type: Optional[str]

    # 최종 답변
    answer: str
    hallucinated: bool  # 환각 여부
    retry_count: int    # 재시도 횟수
# -----------------------
# 2. LLM & 프롬프트 준비
# -----------------------

# 2-1) RAG 필요 여부 + 어떤 소스가 필요한지 라우터용 LLM
router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

router_prompt = ChatPromptTemplate.from_template("""
당신은 RAG 시스템의 라우터입니다.

다음 사용자 질문을 보고, RAG 문서(지역/상권/브랜드/프로모션)가
필요한지 여부와 어떤 종류의 문서가 필요한지를 판단하세요.

질문: {question}

다음 JSON 형식으로만 답변하세요 (설명 없이):

{{
  "need_rag": true or false,
  "rag_sources": ["region_market", "artisee", "promo"] 중 필요한 것만 포함한 리스트
}}

규칙:
- 단순 인사, 잡담, 감사, 스트림릿/환경설정 질문 등은 need_rag=false 로 설정하세요.
- 지역, 상권, 입지, 상권 포함 질문은 "region_market" 을 포함하세요.
- 아티제 톤에 맞춘 전략 설명, 아티제 브랜드 정체성/브랜드 설명 관련 질문은 "artisee" 를 포함하세요.
- 프로모션/이벤트/마케팅/전략 관련 질문은 "promo" 를 포함하세요.
- 하나의 질문이 여러 범주에 걸치면, rag_sources에 모두 포함하세요.
""")

router_chain = router_prompt | router_llm | StrOutputParser()


# 2-2) 최종 답변용 프롬프트 (네가 쓰던 거 기반)
answer_prompt = ChatPromptTemplate.from_template("""
당신은 카페 브랜드 마케팅 및 프로모션 전략 전문가입니다.
사용자 질문, 지역·상권 특성, 브랜드 정보, 프로모션 전략 데이터를 종합해
아티제의 실제 매장 운영에서 적용 가능한 “전략 보고서형 답변”을 작성하세요.

아래 RAG 문서들은 '마케팅 및 프로모션 관련 정보가 필요할 때만' 참고하세요.
일반 대화(인사, 잡담 등)일 경우 RAG 문서를 무시하고 자연스럽게 대화만 하세요.

[지역 기반 인사이트 + 상권 분석]
{local}

[아티제 브랜드 정보]
{artisee}

[카페 프로모션 전략 지식]
{promo}

[지역명: {region}]
[상권 유형: {market_type}]

[이전 대화]
{chat_history}

[사용자 질문]
{question}

---
✦✦✦ 답변 작성 규칙 (특정 상권 또는 지역이 포함된 질문일 때만 사용) ✦✦✦

① **핵심 타겟 고객 정의 (데이터 기반)**
- 지역/상권별 방문 이유, 행동 패턴까지 구체적으로
- “왜 이 고객이 핵심인지” 이유 포함

② **기회 요인**
- 매출 성장 기회 포인트 1~3개도 명시

③ **상권/지역 분석 기반 전략 방향 (Insight → Action 구조, 데이터 기반)**

④ **실행 가능한 프로모션 3~5개 (구체적·수치 포함)**
- 단순 아이디어 X  
- 아래 요소 반드시 포함  
  - 대상 고객  
  - 운영 방식
  - 기대 효과  
  - 이유와 근거

⑤ **추천 이유 요약**
- “왜 이 전략을 선택했는지” 핵심 이유 간략히 정리

⑥  Bullet 구조로 가독성 있게 작성하세요,
  아티제 브랜드 톤/특징 반영해서 작성하세요. 
  [지역 기반 인사이트 + 상권 분석] 가 비어있으면서 전략/프로모션 관련 질문일 경우에는
  이 템플릿을 사용하지 말고 공통 전략 또는 브랜드 차원의 조언만 간단히 제공하세요.


 시간 표현 규칙:
- 시간 범위는 반드시 "~" 기호로 표기 (예: 1시~5시)
- "1시5시", "2시5시", "6시9시"처럼 숫자가 붙으면 안 됨
- "1시-5시", "1시~ 5시" 같은 표기도 금지
- 정확한 형태는 ([숫자]시~[숫자]시)


일반 대화일 경우:
- 위 프로모션 가이드는 무시하고, 자연스러운 톤으로 짧고 친절하게 답변하세요.
""")

main_llm = ChatOpenAI(model="gpt-4o", temperature=0)
answer_chain = answer_prompt | main_llm | StrOutputParser()

def router_node(state: RAGState) -> RAGState:
    """질문을 보고 RAG 필요 여부 + 어떤 소스가 필요한지 판단"""
    q = state["question"]
    raw = router_chain.invoke({"question": q})

    try:
        data = json.loads(raw)
        need_rag = bool(data.get("need_rag", False))
        sources = data.get("rag_sources", [])
        # 방어 코딩: 잘못된 값 들어오면 리스트 비우기
        if not isinstance(sources, list):
            sources = []
    except Exception:
        need_rag = False
        sources = []

    return {
        "need_rag": need_rag,
        "rag_sources": sources,
    }

from langchain.schema import Document

def make_retrieve_node(
    db_region,
    db_market,
    retriever_artisee,
    retriever_promo
):
    strict_retriever = StrictRetriever(db_region=db_region, db_market=db_market)

    def _retrieve(state: RAGState) -> RAGState:
        q = state["question"]
        sources = state.get("rag_sources", [])
        local_text = ""
        artisee_text = ""
        promo_text = ""

        # 1) 지역 + 상권 RAG (StrictRetriever)
        if "region_market" in sources:
            docs_local: list[Document] = strict_retriever.get_relevant_documents(q)
            local_text = "\n\n".join(d.page_content for d in docs_local)

        # 2) 아티제 브랜드 정보
        if "artisee" in sources:
            docs_artisee = retriever_artisee.get_relevant_documents(q)
            artisee_text = "\n\n".join(d.page_content for d in docs_artisee)

        # 3) 프로모션 전략 RAG
        if "promo" in sources:
            docs_promo = retriever_promo.get_relevant_documents(q)
            promo_text = "\n\n".join(d.page_content for d in docs_promo)

        return {
            "local": local_text,
            "artisee": artisee_text,
            "promo": promo_text,
        }

    return _retrieve


def answer_node(state: RAGState) -> RAGState:
    """RAG 결과(또는 빈 문자열) + 질문 + 히스토리로 최종 답변 생성"""
    resp = answer_chain.invoke({
        "question": state["question"],
        "chat_history": state.get("chat_history", ""),
        "local": state.get("local", ""),
        "artisee": state.get("artisee", ""),
        "promo": state.get("promo", ""),
        "region": state.get("region", "") or "",
        "market_type": state.get("market_type", "") or "",
    })
    return {"answer": resp}

def need_hallucination_check(state: RAGState):
    # RAG 안 썼으면 할루 체크 필요 없음
    if state.get("need_rag") is False:
        return "skip"
    return "check"

###################################
# 3. 그래프 빌드 함수
from langgraph.graph import StateGraph, END

def build_rag_graph(
    db_region,
    db_market,
    retriever_artisee,
    retriever_promo,
):
    graph = StateGraph(RAGState)

    # 노드 등록
    graph.add_node("router", router_node)
    graph.add_node("retrieve", make_retrieve_node(
        db_region, db_market, retriever_artisee, retriever_promo
    ))
    graph.add_node("answer", answer_node)


    # ⭐ 새로 추가되는 노드
    graph.add_node("hallucination_check", hallucination_node)

    # 진입점
    graph.set_entry_point("router")

    # router 이후 분기:
    # need_rag == True → retrieve
    # need_rag == False → 바로 answer
    def route_decision(state: RAGState) -> str:
        if state.get("need_rag"):
            return "use_rag"
        return "no_rag"

    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "use_rag": "retrieve",
            "no_rag": "answer",
        }
    )

    # retrieve → answer
    graph.add_edge("retrieve", "answer")

    # answer → hallucination_check
    graph.add_conditional_edges(
        "answer",
        need_hallucination_check,
        {
            "check": "hallucination_check",
            "skip": END
        }
    )

    # hallucination_check 분기
    def hallucination_decision(state: RAGState) -> str:

            # retry 3회 초과 → 더 이상 재시도하지 않고 END
        if state.get("retry_count", 0) >= 3:
            return "ok"

        if state.get("hallucinated"):   # True = 문제가 있음
            return "retry"
        return "ok"

    graph.add_conditional_edges(
        "hallucination_check",
        hallucination_decision,
        {
            "retry": "answer",   # 재생성 노드로 다시 돌아감
            "ok": END,
        }
    )

    app = graph.compile()
    return app



#### Hallucination Grader 준비 (graph_chain.py에 추가)
from pydantic import BaseModel, Field

class GradeHallucinations(BaseModel):
    binary_score: str = Field(
        description="Is the answer grounded in the facts? 'yes' or 'no'"
    )

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system",
    """You are a grader assessing whether an LLM answer is grounded in the retrieved facts.

Respond only in this JSON format:
{{"binary_score": "yes"}}  or  {{"binary_score": "no"}}

"yes" = 답변이 facts에 기반하여 논리적으로 추론된 경우도 포함한다.
"no" = facts와 명백히 모순되거나 사실과 다른 내용인 경우.
- "추론(inference)" 또는 "전략 제안(strategy proposal)"은 hallucination이 아니다.


"""),

    ("human",
    "Facts:\n{documents}\n\nLLM Answer:\n{generation}")
])


hallucination_grader = hallucination_prompt | structured_llm_grader

def hallucination_node(state: RAGState) -> RAGState:

    context = "\n\n".join([
        state.get("local", ""),
        state.get("artisee", ""),
        state.get("promo", "")
    ])

    grade: GradeHallucinations = hallucination_grader.invoke({
        "documents": context,
        "generation": state["answer"]
    })

    hallucinated = (grade.binary_score.lower() == "no")

    retry = state.get("retry_count", 0)

    return {
        "hallucinated": hallucinated,
        "retry_count": retry + 1 if hallucinated else retry
    }




