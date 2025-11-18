from langchain_openai import ChatOpenAI
import re
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def pick_best_region(region, db_region):
    """상위 지역(인천/수원 등)을 DB 안의 실제 지역명으로 자동 매핑"""
    if region is None:
        return None

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


import re
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

###############################################
# 1) 서울 구 자동 정규화 딕셔너리
###############################################
SEOUL_DISTRICTS = [
    "강남구", "서초구", "송파구", "광진구", "강동구", "중구", "용산구", "종로구",
    "마포구", "서대문구", "영등포구", "동대문구", "양천구", "동작구", "강서구"
]

def normalize_seoul(region: str):
    """ '서울 강남구', '서울시 강남' → '강남구' 로만 반환 """
    if region is None:
        return None
    for dist in SEOUL_DISTRICTS:
        # 강남, 서초 등 '구' 생략된 형태도 잡음
        if dist.replace("구", "") in region:
            return dist
    return region


###############################################
# 2) LLM 출력 파싱용 정규식 개선 (":", "=", 공백, 따옴표 전부 허용)
###############################################
def parse_region_market_from_response(response: str):
    region_match = re.search(r"region\s*[:=]\s*[\"']?([\w가-힣\s]+)[\"']?", response)
    market_match = re.search(r"market_type\s*[:=]\s*[\"']?([\w가-힣\s\(\)]+)[\"']?", response)

    region = region_match.group(1).strip() if region_match else None
    market = market_match.group(1).strip() if market_match else None

    return region, market





###############################################
# 4) 메인 함수 — LLM → 정규화 → 벡터DB 매핑까지
###############################################
def extract_region_market(question):
    prompt = f"""
    아래 질문에서 '지역명'과 '상권명'을 정확히 추출해줘.

    지역 추출 규칙:
    - 'OO구', 'OO시', 'OO', 'OO광역시' 모두 인정
    - '서울 강남구' → '강남구'
    - '서울시 강남구' → '강남구'
    - '대구', '인천'처럼 시 단위면 대표 구로 해석
      * 대구광역시 → 대구 북구
      * 인천광역시 → 인천 연수구

    출력 형식은 아래만 사용해:
    region=<지역명>
    market_type=<상권명>

    질문: {question}
    """

    response = llm.invoke(prompt).content

    # ① region / market 파싱
    region, market_type = parse_region_market_from_response(response)

    # ② 서울 구 자동 정규화
    region = normalize_seoul(region)

    # ③ Vector DB 기반 최종 매칭
    emb = OpenAIEmbeddings(model="text-embedding-3-large")
    db_region = Chroma(
        persist_directory="./rag_region6",
        embedding_function=emb
    )
    region = pick_best_region(region, db_region)

    return region, market_type

# def extract_region_market(question):
#     prompt = f"""
#     질문 문장에서 지역명과 상권명을 정확히 추출해줘.

#     지역 추출 규칙:
#     - 'OO구', 'OO시', 'OO', 'OO광역시' 모두 인정
#     - '대구', '인천', '고양'처럼 시 단위면 해당 시의 대표 구로 해석
#     - 대구광역시 → 대구 북구
#     - 인천광역시 → 인천 연수구

#     출력:
#     region=<지역명>, market_type=<상권명>

#     질문: {question}
#     """
#     response = llm.invoke(prompt).content
    
#     region_match = re.search(r"region=([\w가-힣\s]+)", response)
#     market_match = re.search(r"market_type=([\w가-힣\s\(\)]+)", response)

#     region = region_match.group(1).strip() if region_match else None
#     market_type = market_match.group(1).strip() if market_match else None

#     # region 자동 매핑(인천 → 인천 연수구)
#     emb = OpenAIEmbeddings(model="text-embedding-3-large")

#     db_region = Chroma(
#         persist_directory="./rag_region6",
#         embedding_function=emb
#     )
#     region = pick_best_region(region, db_region)

#     return region, market_type

#     - 서울시 안에 강남구, 서초구, 송파구, 광진구, 강동구, 중구, 용산구, 종로구, 마포구, 서대문구, 영등포구, 동대문구, 양천구, 동작구, 강서구, 종로구가 포함되어 있음.
