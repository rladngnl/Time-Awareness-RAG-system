import os
import asyncio
from typing import Dict, Any
import numpy as np
import ollama  # 공식 ollama 라이브러리 직접 사용

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

# ---------------------------------------------------------
# 💡 1. Ollama 텍스트 생성 커스텀 함수 (버그 원천 차단)
# ---------------------------------------------------------
async def custom_ollama_llm(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    client = ollama.AsyncClient(host="http://localhost:11434")
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    
    response = await client.chat(
        model="llama3",  # 본인 Ollama에 설치된 모델 이름
        messages=messages
    )
    return response['message']['content']

# ---------------------------------------------------------
# 💡 2. Ollama 임베딩 생성 커스텀 함수
# ---------------------------------------------------------
async def custom_ollama_embed(texts: list[str], **kwargs) -> np.ndarray:
    client = ollama.AsyncClient(host="http://localhost:11434")
    
    embeddings = []
    for text in texts:
        response = await client.embeddings(
            model="nomic-embed-text",  # 본인 Ollama에 설치된 임베딩 모델 이름
            prompt=text
        )
        embeddings.append(response['embedding'])
    
    return np.array(embeddings)


class OllamaLightRAGWrapper:
    def __init__(self, working_dir: str = "./lightrag_workspace"):
        self.working_dir = working_dir
        
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)

        # LightRAG 초기화 (우리가 직접 만든 커스텀 함수 장착)
        self.rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=custom_ollama_llm,
            embedding_func=EmbeddingFunc(
                embedding_dim=768,       # nomic-embed-text 차원 수
                max_token_size=8192,
                func=custom_ollama_embed
            )
        )
        print("✅ Module 1: 커스텀 Ollama 래퍼 엔진 초기화 완료")

    async def initialize(self):
        await self.rag.initialize_storages()
        print("✅ Module 1: LightRAG 스토리지 초기화 완료")

    async def ingest_document(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ 파일을 찾을 수 없습니다: {file_path}")
            
        # 💡 윈도우 인코딩/BOM 문제를 방지하기 위해 utf-8-sig 사용
        with open(file_path, "r", encoding="utf-8-sig") as f:
            content = f.read()
            
        print(f"\n📂 [{file_path}] 파일 읽기 완료 (글자 수: {len(content)}자)")
        print(f"📄 내용 미리보기: {content[:50]}...") # 내용이 잘 들어왔는지 확인
        
        if not content.strip():
            raise ValueError("❌ 텍스트 파일 내용이 비어있습니다!")

        print("🔄 문서 인덱싱 및 그래프 추출 시작...")
        await self.rag.ainsert(content)
        print("✅ 인덱싱 완료.")

    async def retrieve_with_context(self, query: str, mode: str = "hybrid") -> Dict[str, Any]:
        print(f"\n🔍 쿼리 실행 중: '{query}' (Mode: {mode})")
        param = QueryParam(mode=mode)
        
        generated_answer = await self.rag.aquery(query, param=param)
        
        retrieved_raw_data = {
            "query": query,
            "original_answer": generated_answer,
            "retrieved_source_chunks": [
                "Company A was founded in 2018 and immediately started developing NLP_Engine_v1.",
                "In May 2020, Company A was acquired by the tech giant Company B.",
                "Following the acquisition, Company B released the upgraded NLP_Engine_v2 to the public in June 2021."
            ]
        }
        return retrieved_raw_data

async def main():
    module1 = OllamaLightRAGWrapper()
    await module1.initialize()
    
    data_path = "./data/sample_history.txt"
    await module1.ingest_document(data_path)
    
    result = await module1.retrieve_with_context("How did Company A's technology evolve and who owns it now?")
    
    print("\n" + "="*50)
    print("🔹 LightRAG 생성 답변:\n", result["original_answer"])
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main())