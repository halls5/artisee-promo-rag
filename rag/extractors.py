from langchain_openai import ChatOpenAI
import re
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def pick_best_region(region, db_region):
    """상위 지역(인천/수원 등)을 DB 안의 실제 지역명으로 자동 매핑"""


    collection = db_region._collection.get()
    all_regions = [
        m.get("region")
        for m in collection["metadatas"]
        if m.get("region") is not None
    ]

    # 1) 정확히 일치하면 그대로 사용
    if region in all_regions:
        return region

    # 2) 부분 매칭 (인천 -> 인천 연수구, 수원 -> 수원시 영통구)
    candidates = [r for r in all_regions if region in r]
    if candidates:
        # 가장 긴 문자열 선택 (더 구체적인 행정동일수록 좋음)
        return max(candidates, key=len)

    # 3) fallback: 그래도 없으면 원래 region 반환
    return region



def extract_region_market(question):
    prompt = f"""
    질문 문장에서 지역명과 상권명을 정확히 추출해줘.

    지역 추출 규칙:
    - 'OO구', 'OO시', 'OO', 'OO광역시' 모두 인정
    - '대구', '인천', '고양'처럼 시 단위면 해당 시의 대표 구로 해석
    - 대구광역시 → 대구 북구
    - 인천광역시 → 인천 연수구

    출력:
    region=<지역명>, market_type=<상권명>

    질문: {question}
    """
    response = llm.invoke(prompt).content
    
    region_match = re.search(r"region=([\w가-힣\s]+)", response)
    market_match = re.search(r"market_type=([\w가-힣\s\(\)]+)", response)

    region = region_match.group(1).strip() if region_match else None
    market_type = market_match.group(1).strip() if market_match else None

    # region 자동 매핑(인천 → 인천 연수구)
    emb = OpenAIEmbeddings(model="text-embedding-3-large")

    db_region = Chroma(
        persist_directory="./rag_region6",
        embedding_function=emb
    )
    region = pick_best_region(region, db_region)

    return region, market_type

# def extract_region_market(question):
#     prompt = f"""
#     다음 문장에서 지역명과 상권명을 정확히 추출해줘.

#  지역 추출 규칙:
# - 'OO구', 'OO시', 'OO광역시' 모두 지역으로 인정
# - '강남', '서초', '수원', '대구'처럼 끝에 행정구역이 없어도 
#   실제 존재하는 지역으로 보정해서 반환
#   예: 강남 → 강남구, 대구 → 대구 북구
# - 문장에 '대구 북구'처럼 더 구체적인 지역이 있으면 그걸 우선 선택
# '상권명(예: 오피스상권, 병원상권, 주택가상권, 복합상권, 쇼핑몰상권, 위탁점포)'을 각각 추출해줘.
# 존재하지 않으면 None으로 표시해.
# 질문: {question}
# 출력형식: region=<지역명>, market_type=<상권명>
# """
#     response = llm.invoke(prompt).content
#     region_match = re.search(r"region=([\w가-힣\s]+)", response)
#     market_match = re.search(r"market_type=([\w가-힣\s\(\)]+)", response)

#     region = region_match.group(1) if region_match else None
#     market_type = market_match.group(1) if market_match else None
    
#     return region, market_type
