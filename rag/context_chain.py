from operator import itemgetter
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from rag.extractors import extract_region_market
from rag.retrievers import StrictRetriever

def build_chain(db_region, db_market, retriever_artisee, retriever_promo, llm):

    retriever = StrictRetriever(db_region=db_region, db_market=db_market)

    # 1. multi-context 구성
    multi_context = RunnableParallel(
        # StrictRetriever
        local=itemgetter("question") | RunnableLambda(
            lambda q: retriever.get_relevant_documents(q)
        ),

        # 아티제 정보
        artisee=itemgetter("question") | RunnableLambda(
            lambda q: retriever_artisee.get_relevant_documents(q)
        ),

        # 프로모션 DB
        promo=itemgetter("question") | RunnableLambda(
            lambda q: retriever_promo.get_relevant_documents(q)
        ),

        # 그대로 전달되는 값들
        question=itemgetter("question"),

        region=itemgetter("question") | RunnableLambda(lambda q: extract_region_market(q)[0]),
        market_type=itemgetter("question") | RunnableLambda(lambda q: extract_region_market(q)[1]),

        chat_history=itemgetter("chat_history")
    )



    # 2. prompt
    prompt = ChatPromptTemplate.from_template("""
당신은 카페 브랜드 마케팅 및 프로모션 전략 전문가입니다.

아래는 사용자 질문을 위해 검색된 3개의 데이터 소스입니다.

아래 RAG 문서들은 {question}이 '마케팅 및 프로모션 관련 질문일 때만 정보가 필요할 때만' 사용하세요.
프로모션과 무관한 질문이라면 RAG 문서를 절대 사용하지 말고, 일반 대화로 답변하세요.

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

--- 프로모션 추천 출력 가이드 ---
1) 핵심 타겟 고객 정리  
2) 지역/상권 특성 기반 프로모션 방향  
3) 아티제 브랜드 톤에 맞춘 실행 아이디어  
4) 질문에 계절·날씨 요인 포함 시 반영  
5) 실제 실행 가능한 프로모션 3~5개  
6) 추천 이유 포함
""")

    # 3. 최종 chain
    chain = multi_context | prompt | llm | StrOutputParser()
    return chain
