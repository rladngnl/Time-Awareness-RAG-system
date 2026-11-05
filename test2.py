import os
import asyncio
from typing import Dict, Any
from lightrag import LightRAG, QueryParam
from lightrag.llm import ollama_model_complete, ollama_embedding

class OllamaLightRAGWrapper:
    """
    Module 1: Ollama 기반 순정 LightRAG 엔진 래퍼
    모든 LLM 추론과 임베딩 생성을 로컬 Ollama를 통해서만 수행하도록 설정합니다.
    """
    def __init__(
        self, 
        working_dir: str = "./lightrag_workspace",
        llm_model: str = "llama3:8b",          # 텍스트 생성용 모델 (Ollama에 설치된 이름)
        embedding_model: str = "nomic-embed-text" # 임베딩용 모델 (Ollama에 설치 권장)
    ):
        self.working_dir = working_dir
        self.llm_model = llm_model
        
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
            
        # Ollama LLM 호출 함수를 LightRAG 스펙에 맞게 바인딩
        async def ollama_llm_func(prompt, **kwargs):
            return await ollama_model_complete(
                prompt=prompt,
                model_name=self.llm_model,
                host="http://localhost:11434", # Ollama 기본 포트
                **kwargs
            )
            
        # Ollama Embedding 호출 함수 바인딩
        async def ollama_embed_func(texts, **kwargs):
            return await ollama_embedding(
                texts=texts,
                model_name=embedding_model,
                host="http://localhost:11434",
                **kwargs
            )

        # LightRAG 인스턴스 초기화 (Ollama 주입)
        self.rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=ollama_llm_func,
            llm_model_max_token_size=4096, # 로컬 모델 컨텍스트 길이에 맞게 조정
            embedding_func=ollama_embed_func,
            embedding_batch_num=32,        # 메모리에 맞춰 배치 크기 조절
            embedding_func_max_tokens=4096,
        )
        print(f"Module 1: Ollama LightRAG 엔진 초기화 완료 (LLM: {llm_model}, Embed: {embedding_model})")

    async def ingest_document(self, file_path: str):
        """문서를 읽어서 지식 그래프를 생성합니다."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        print(f"[{file_path}] 문서 인덱싱 시작... (Ollama 동작 중)")
        await self.rag.ainsert(content)
        print("인덱싱 완료.")

    async def retrieve_with_context(self, query: str, mode: str = "hybrid") -> Dict[str, Any]:
        """
        검색(Retrieval)을 수행하고, Module 2(Temporal Adapter)가 시간 역추적에 
        사용할 수 있도록 검색된 원본 텍스트 청크를 함께 확보하여 반환합니다.
        """
        param = QueryParam(mode=mode)
        
        # 1. 쿼리 실행 (Ollama가 그래프를 탐색하여 1차 답변 생성)
        generated_answer = await self.rag.aquery(query, param=param)
        
        # 2. Module 2를 위한 컨텍스트 추출 (시뮬레이션)
        # 실제 환경에서는 LightRAG의 워크스페이스 내 kv_store.json이나 
        # graph_chunk_entity_relation.graphml 파일을 파싱하여 쿼리에 매칭된 원문을 가져와야 합니다.
        retrieved_raw_data = {
            "query": query,
            "original_answer": generated_answer,
            # Module 2의 Time Extractor가 읽게 될 원시 텍스트
            "retrieved_source_chunks": [
                "Company A developed NLP_Engine_v1 from 2018 to 2020.",
                "In May 2020, Company A was acquired by Company B."
            ]
        }
        
        return retrieved_raw_data

# --- 사용 방법 ---
# asyncio.run(main()) 을 통해 실행할 수 있습니다.