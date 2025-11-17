def create_retrievers(db_region, db_market, db_artisee, db_promo):
    retriever_region = db_region.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1}
    )

    retriever_market = db_market.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1}
    )

    retriever_artisee = db_artisee.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # retriever_promo = db_promo.as_retriever(
    #     search_type="similarity",
    #     search_kwargs={"k": 5}
    # )
    retriever_promo_vector = db_promo.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
    )

    retriever_promo = PromoRetriever(
        vector_retriever=retriever_promo_vector,
        db_promo=db_promo
    )


    return retriever_region, retriever_market, retriever_artisee, retriever_promo

from langchain.schema import BaseRetriever, Document
from langchain_core.pydantic_v1 import Field
from rag.extractors import extract_region_market


class StrictRetriever(BaseRetriever):
    db_region: any = Field(...)
    db_market: any = Field(...)

    def _get_relevant_documents(self, query):
        """LangChain RAG 내부용 실제 검색 로직"""

        region, market_type = extract_region_market(query)

        # 1) 지역 필터로 지역 리뷰 찾기
        docs_region = self.db_region._collection.get(where={"region": region})

        # 2) 상권 유형으로 시장 정보 찾기
        docs_market = self.db_market._collection.get(where={"market_type": market_type})

        # Document 객체로 변환
        results = []

        if docs_region and "documents" in docs_region:
            for d in docs_region["documents"]:
                results.append(
                    Document(
                        page_content=d,
                        metadata={"region": region}
                    )
                )

        if docs_market and "documents" in docs_market:
            for d in docs_market["documents"]:
                results.append(
                    Document(
                        page_content=d,
                        metadata={"market_type": market_type}
                    )
                )

        return results

    async def _aget_relevant_documents(self, query):
        """비동기 버전 (동기 로직 재사용)"""
        return self._get_relevant_documents(query)

class PromoRetriever(BaseRetriever):
    vector_retriever: any = Field(...)
    db_promo: any = Field(...)

    def _get_relevant_documents(self, query):

        case_docs = self.vector_retriever.get_relevant_documents(query)

        common_raw = self.db_promo._collection.get(where={"type": "common"})
        common_docs = []

        if common_raw and "documents" in common_raw:
            for d in common_raw["documents"]:
                common_docs.append(
                    Document(page_content=d, metadata={"type": "common"})
                )

        return common_docs + case_docs
