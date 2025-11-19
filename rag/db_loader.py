from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

def load_all_dbs():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    db_region = Chroma(
        persist_directory="./rag_region6",
        embedding_function=embeddings
    )

    db_market = Chroma(
        persist_directory="./rag_market",
        embedding_function=embeddings
    )

    db_artisee = Chroma(
        persist_directory="./rag_artisee_info2",
        embedding_function=embeddings
    )

    db_promo = Chroma(
        persist_directory="./rag_promo2",
        embedding_function=embeddings
    )

    return db_region, db_market, db_artisee, db_promo

import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.dirname(BASE_DIR)   # 12-스트림릿 폴더 기준 상위경로

# def load_all_dbs():
#     embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

#     db_region = Chroma(
#         persist_directory=os.path.join(ROOT_DIR, "rag_region5"),
#         embedding_function=embeddings
#     )

#     db_market = Chroma(
#         persist_directory=os.path.join(ROOT_DIR, "rag_market3"),
#         embedding_function=embeddings
#     )

#     db_artisee = Chroma(
#         persist_directory=os.path.join(ROOT_DIR, "rag_artisee_info3"),
#         embedding_function=embeddings
#     )

#     db_promo = Chroma(
#         persist_directory=os.path.join(ROOT_DIR, "rag_promo3"),
#         embedding_function=embeddings
#     )

#     return db_region, db_market, db_artisee, db_promo

