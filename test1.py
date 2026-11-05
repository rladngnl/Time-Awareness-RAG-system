import os
import asyncio
from lightrag import LightRAG
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc

# ==========================================
# 1. 래퍼(Wrapper) 클래스 정의 (llama3 적용)
# ==========================================
class TemporalLightRAGWrapper:
    def __init__(self, working_dir="./tkg_workspace"):
        # 임베딩은 그대로 nomic-embed-text 유지
        async def local_embed(texts: list[str], **kwargs):
            kwargs["model"] = "nomic-embed-text" 
            return await ollama_embed(texts, **kwargs)

        self.rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=ollama_model_complete,
            llm_model_name="llama3",  # 👈 qwen2.5에서 llama3로 변경!
            embedding_func=EmbeddingFunc(
                embedding_dim=768,
                max_token_size=8192,
                func=local_embed,
            ),
        )

    async def initialize(self):
        """파이프라인 필수 초기화"""
        await self.rag.initialize_storages()

    async def index_historical_fact(self, target_entity: str, fact_text: str, timestamp: str):
        """
        과거 사실(Fact)에 시간 메타데이터를 주입하여 인덱싱합니다.
        """
        # 시간 정보를 명시적인 문맥 꼬리표(Lexical Anchor)로 텍스트화
        time_aware_text = f"[[기록 대상: {target_entity} | 발생 시점: {timestamp}]]\n내용: {fact_text}"
        
        print(f"[{timestamp}] '{target_entity}'의 데이터를 처리 중입니다...")
        await self.rag.ainsert(time_aware_text)
        print(f"✅ '{target_entity}' 데이터 인덱싱 완료!\n")

# ==========================================
# 2. 실행 테스트 코드
# ==========================================
async def main():
    # 작업 폴더 지정
    wrapper = TemporalLightRAGWrapper(working_dir="./tkg_workspace_llama3")
    await wrapper.initialize()

    print("--- TKG 히스토리 인덱싱 시작 (LLM: llama3) ---\n")

    # 가상의 인물 '김철수'의 히스토리 데이터 주입
    await wrapper.index_historical_fact(
        target_entity="김철수",
        fact_text="김철수는 삼성전자에 신입 사원으로 입사하여 스마트폰 사업부에서 근무를 시작했다.",
        timestamp="2021-03-01"
    )

    await wrapper.index_historical_fact(
        target_entity="김철수",
        fact_text="김철수는 삼성전자를 퇴사하고, 카카오 AI 연구소로 이직하여 시니어 연구원으로 발령받았다.",
        timestamp="2023-08-15"
    )

    await wrapper.index_historical_fact(
        target_entity="김철수",
        fact_text="김철수는 카카오에서 독립하여 AI 스타트업 '미래테크'를 창업하고 CEO로 취임했다.",
        timestamp="2025-01-10"
    )

    print("--- 모든 히스토리 데이터 인덱싱 완료 ---")

if __name__ == "__main__":
    asyncio.run(main())