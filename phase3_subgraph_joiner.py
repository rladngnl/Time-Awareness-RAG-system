import os
import json
import asyncio
from typing import Dict, Any, List
import numpy as np
import ollama

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

# ---------------------------------------------------------
# 1. LightRAG 구동용 커스텀 Ollama 함수 (임베딩 및 키워드 추출용)
# ---------------------------------------------------------
async def custom_ollama_llm(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    client = ollama.AsyncClient(host="http://localhost:11434")
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    
    response = await client.chat(model="llama3", messages=messages)
    return response['message']['content']

async def custom_ollama_embed(texts: list[str], **kwargs) -> np.ndarray:
    client = ollama.AsyncClient(host="http://localhost:11434")
    embeddings = []
    for text in texts:
        response = await client.embeddings(model="nomic-embed-text", prompt=text)
        embeddings.append(response['embedding'])
    return np.array(embeddings)


# ---------------------------------------------------------
# 2. Phase 3: 시계열 서브그래프 결합기 (Temporal Subgraph Joiner)
# ---------------------------------------------------------
class TemporalSubgraphJoiner:
    """
    Phase 3: 사용자 쿼리에 대해 LightRAG가 검색한 서브그래프 컨텍스트를 가로채고,
    이를 Shadow Temporal Index와 결합(Join)하여 Chronological Cypher Table을 생성합니다.
    """
    def __init__(
        self,
        lightrag_dir: str = "./lightrag_workspace",
        shadow_dir: str = "./shadow_workspace"
    ):
        self.lightrag_dir = lightrag_dir
        self.shadow_dir = shadow_dir
        
        self.temporal_index_path = os.path.join(shadow_dir, "temporal_shadow_index.json")
        self.output_json_path = os.path.join(shadow_dir, "phase3_joined_context.json")
        self.output_txt_path = os.path.join(shadow_dir, "phase3_cypher_table_report.txt")

        # LightRAG 인스턴스 연결 (기존 인덱싱된 워크스페이스 로드)
        self.rag = LightRAG(
            working_dir=self.lightrag_dir,
            llm_model_func=custom_ollama_llm,
            embedding_func=EmbeddingFunc(
                embedding_dim=768,
                max_token_size=8192,
                func=custom_ollama_embed
            )
        )

    async def initialize(self):
        await self.rag.initialize_storages()
        print("✅ Phase 3: LightRAG 스토리지 연결 완료")

    def load_temporal_index(self) -> Dict[str, Any]:
        if not os.path.exists(self.temporal_index_path):
            raise FileNotFoundError(f"❌ Phase 2 시간축 DB를 찾을 수 없습니다: {self.temporal_index_path}")
        with open(self.temporal_index_path, "r", encoding="utf-8") as f:
            return json.load(f)

    async def intercept_and_join(self, query: str, mode: str = "hybrid") -> Dict[str, Any]:
        print(f"\n🔍 [쿼리 실행] '{query}' (Mode: {mode})")
        print("⏳ LightRAG로부터 서브그래프 컨텍스트 가로채기(Intercept) 수행 중...")
        
        # 💡 핵심: only_need_context=True로 설정하여 LLM 답변 생성을 생략하고 검색된 서브그래프만 획득
        param = QueryParam(mode=mode, only_need_context=True)
        raw_context = await self.rag.aquery(query, param=param)
        
        # 시간축 DB 로드
        temporal_index = self.load_temporal_index()
        
        # 검색된 컨텍스트 내부에 등장하는 엣지들과 Shadow Temporal Index 결합(Join)
        matched_edges = []
        raw_context_lower = raw_context.lower() if raw_context else ""

        for lookup_key, edge_record in temporal_index.items():
            src = edge_record["source"]
            tgt = edge_record["target"]
            
            # 검색된 서브그래프 컨텍스트 내에 Source 노드와 Target 노드가 모두 포함되어 있는지 확인
            if src.lower() in raw_context_lower and tgt.lower() in raw_context_lower:
                matched_edges.append(edge_record)

        print(f"✅ 서브그래프 Join 완료: 총 {len(matched_edges)}개의 시계열 엣지가 매칭되었습니다.")

        # 시작 시간(start_time) 기준 오름차순 정렬 ('Unknown'은 가장 뒤로 배치)
        sorted_edges = sorted(
            matched_edges,
            key=lambda x: (x["start_time"] == "Unknown", x["start_time"])
        )

        # Chronological Cypher Table 포맷 렌더링
        cypher_table = self.build_chronological_cypher_table(sorted_edges)

        result_payload = {
            "query": query,
            "retrieval_mode": mode,
            "matched_edge_count": len(sorted_edges),
            "chronological_cypher_table": cypher_table,
            "sorted_temporal_edges": sorted_edges,
            "raw_lightrag_context": raw_context
        }
        return result_payload

    def build_chronological_cypher_table(self, sorted_edges: List[Dict[str, Any]]) -> str:
        """정렬된 시계열 엣지 리스트를 LLM 친화적인 Chronological Cypher Table 문자열로 변환합니다."""
        if not sorted_edges:
            return "No temporal graph edges matched for this query."

        lines = [
            "| Timestamp | Cypher Relation Pattern | Relation Details & Temporal Note |",
            "| :--- | :--- | :--- |"
        ]
        
        for edge in sorted_edges:
            start = edge["start_time"]
            end = edge["end_time"]
            time_str = start if start == end else f"{start} ~ {end}"
            
            # 관계 키워드 정리 (첫 번째 대표 키워드를 Cypher 엣지 라벨로 포맷팅)
            raw_kw = edge["keywords"].split(",")[0].strip().upper().replace(" ", "_")
            if not raw_kw:
                raw_kw = "RELATED_TO"
                
            cypher_pattern = f"({edge['source']})-[:{raw_kw}]->({edge['target']})"
            details = f"{edge['relation_desc']} (Note: {edge['temporal_note']})"
            
            lines.append(f"| {time_str} | {cypher_pattern} | {details} |")

        return "\n".join(lines)

    def save_results(self, payload: Dict[str, Any]):
        """결합된 컨텍스트 결과를 JSON(Phase 4 입력용)과 TXT(리포트용)로 저장합니다."""
        with open(self.output_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=4)

        with open(self.output_txt_path, "w", encoding="utf-8") as f:
            f.write("=" * 85 + "\n")
            f.write("🔗 [Phase 3 결과 리포트] 서브그래프 가로채기 및 시간축 결합(Join) 결과\n")
            f.write("=" * 85 + "\n\n")
            f.write(f"• 사용자 질문 (Query): {payload['query']}\n")
            f.write(f"• 검색 모드 (Mode): {payload['retrieval_mode']}\n")
            f.write(f"• 매칭된 시계열 엣지 수: {payload['matched_edge_count']}개\n\n")
            
            f.write("-" * 85 + "\n")
            f.write("### 1. 생성된 Chronological Cypher Table (Phase 4 Generator 입력용)\n")
            f.write("-" * 85 + "\n")
            f.write(payload["chronological_cypher_table"] + "\n\n")
            
            f.write("-" * 85 + "\n")
            f.write("### 2. [참고] LightRAG로부터 가로챈 원본 서브그래프 컨텍스트 (Raw Context)\n")
            f.write("-" * 85 + "\n")
            f.write(str(payload["raw_lightrag_context"]) + "\n")

        print(f"\n💾 Phase 3 결과 저장 완료!")
        print(f"  👉 JSON 데이터 (Phase 4 연동용): {self.output_json_path}")
        print(f"  👉 TXT 리포트 (테이블 확인용): {self.output_txt_path}")


# === 실행부 ===
async def main():
    joiner = TemporalSubgraphJoiner()
    await joiner.initialize()
    
    # 시계열 추론이 필요한 테스트 쿼리 실행
    test_query = "How did Company A's technology evolve over time and who owns it now?"
    result_payload = await joiner.intercept_and_join(test_query, mode="hybrid")
    
    joiner.save_results(result_payload)
    
    print("\n" + "=" * 85)
    print("📊 [생성된 Chronological Cypher Table 미리보기]")
    print("=" * 85)
    print(result_payload["chronological_cypher_table"])
    print("=" * 85)

if __name__ == "__main__":
    asyncio.run(main())