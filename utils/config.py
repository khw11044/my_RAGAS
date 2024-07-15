"""
PATH, URL 등 전역 상수 설정
"""
# 필요시 클래스로 선언

# intfloat/multilingual-e5-small
config = {
    "llm_predictor": {
        "model_name": "gpt-3.5-turbo",
        "temperature": 0
    },
    "embed_model": {
        "model_name": "text-embedding-ada-002",
        "cache_directory": "",
    },
    "chroma": {
        "persist_dir": "./database",
    },
    "path": {
        "input_directory": "./documents",
    },
    "search_type": "similarity",
    "ensemble_search_type": "mmr",
    "similarity_k": 0.25,
    "retriever_k": 5,           # 유사도 기반 가장 유사한 top 5개 청크 가져오기 
}
