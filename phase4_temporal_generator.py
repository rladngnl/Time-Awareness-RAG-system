import os
import json
import asyncio
from typing import Dict, Any
import ollama

class TemporalSFTGenerator:
    """
    Phase 4: Phase 3에서 결합된 Chronological Cypher Table을 입력받아,
    시간적 선후 관계와 상태 변화(State Transition)를 추론하여 최종 답변을 생성합니다.
    (연구 비교를 위해 일반 LightRAG 컨텍스트 기반의 Baseline 답변도 함께 생성합니다.)
    """
    def __init__(
        self,
        shadow_dir: str = "./shadow_workspace",
        llm_model: str = "llama3"  # 추후 파인튜닝된 SFT 모델 이름으로 교체 가능
    ):
        self.shadow_dir = shadow_dir
        self.llm_model = llm_model
        
        self.input_json_path = os.path.join(shadow_dir, "phase3_joined_context.json")
        self.output_json_path = os.path.join(shadow_dir, "phase4_final_output.json")
        self.output_txt_path = os.path.join(shadow_dir, "phase4_final_answer_report.txt")
        
        self.client = ollama.AsyncClient(host="http://localhost:11434")

    def load_phase3_context(self) -> Dict[str, Any]:
        if not os.path.exists(self.input_json_path):
            raise FileNotFoundError(f"❌ Phase 3 결과 파일을 찾을 수 없습니다: {self.input_json_path}")
        with open(self.input_json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    async def generate_temporal_answer(self, query: str, cypher_table: str) -> str:
        """[Proposed] Chronological Cypher Table을 기반으로 시계열 추론 답변을 생성합니다."""
        
        system_prompt = """You are an advanced Temporal Knowledge Graph Reasoning AI.
You are provided with a 'Chronological Cypher Table' where graph edges (relationships) are strictly sorted in chronological order from past to present.

Follow these steps to answer the user's question:
1. [Temporal Timeline Analysis]: Read the table from top to bottom. Explicitly trace the timeline of events and state transitions (e.g., creation, completion, ownership transfer, version upgrade).
2. [Current State Verification]: Identify the most recent timestamp to determine the current valid state (e.g., current owner or latest technology).
3. [Final Synthesis]: Provide a clear, temporally consistent answer without any chronological contradictions."""

        user_prompt = f"""### Chronological Cypher Table (Sorted by Time)
{cypher_table}

### User Question
{query}

Please provide your response structured with:
1. **Chronological Reasoning (Step-by-Step Timeline)**
2. **Final Answer**"""

        response = await self.client.chat(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={"temperature": 0.1}  # 사실 기반 추론을 위해 낮은 온도 설정
        )
        return response['message']['content']

    async def generate_baseline_answer(self, query: str, raw_context: str) -> str:
        """[Baseline 비교용] 시간 정렬 테이블 없이 순정 LightRAG의 Raw Context만으로 답변을 생성합니다."""
        
        system_prompt = """You are a helpful AI assistant. Answer the user's question based on the provided context."""
        
        user_prompt = f"""### Retrieved Context
{raw_context}

### User Question
{query}

Answer:"""

        response = await self.client.chat(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={"temperature": 0.1}
        )
        return response['message']['content']

    async def run_generation_pipeline(self) -> Dict[str, Any]:
        payload = self.load_phase3_context()
        
        query = payload["query"]
        cypher_table = payload["chronological_cypher_table"]
        raw_context = payload.get("raw_lightrag_context", "")

        print(f"📂 Phase 3 컨텍스트 로드 완료 (매칭된 시계열 엣지: {payload['matched_edge_count']}개)")
        print(f"🔍 질문: '{query}'")
        
        print("\n⏳ 1. [Proposed] Chronological Cypher Table 기반 시계열 추론 답변 생성 중...")
        temporal_answer = await self.generate_temporal_answer(query, cypher_table)
        print("   ✅ 시계열 추론 답변 생성 완료!")

        print("⏳ 2. [Baseline] 일반 LightRAG Raw Context 기반 비교 답변 생성 중...")
        baseline_answer = await self.generate_baseline_answer(query, raw_context)
        print("   ✅ Baseline 답변 생성 완료!")

        final_output = {
            "query": query,
            "matched_edge_count": payload["matched_edge_count"],
            "chronological_cypher_table": cypher_table,
            "proposed_temporal_answer": temporal_answer,
            "baseline_raw_answer": baseline_answer
        }
        return final_output

    def save_results(self, final_output: Dict[str, Any]):
        """최종 생성 결과를 JSON과 비교 리포트 TXT 파일로 저장합니다."""
        
        # 1. JSON 저장
        with open(self.output_json_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, ensure_ascii=False, indent=4)

        # 2. 비교 리포트 TXT 저장
        with open(self.output_txt_path, "w", encoding="utf-8") as f:
            f.write("=" * 90 + "\n")
            f.write("🏆 [Phase 4 최종 리포트] Shadow TKG-RAG 시계열 추론 결과 및 Baseline 비교\n")
            f.write("=" * 90 + "\n\n")
            
            f.write(f"❓ [User Query]: {final_output['query']}\n\n")
            
            f.write("-" * 90 + "\n")
            f.write("📊 1. 입력된 Chronological Cypher Table (Shadow Wrapper 출력물)\n")
            f.write("-" * 90 + "\n")
            f.write(final_output["chronological_cypher_table"] + "\n\n")
            
            f.write("=" * 90 + "\n")
            f.write("🚀 2. [Proposed] Shadow TKG-RAG 최종 답변 (Cypher Table 기반 추론)\n")
            f.write("=" * 90 + "\n")
            f.write(final_output["proposed_temporal_answer"] + "\n\n")
            
            f.write("=" * 90 + "\n")
            f.write("🐢 3. [Baseline] 순정 LightRAG 컨텍스트 기반 답변 (비교군)\n")
            f.write("=" * 90 + "\n")
            f.write(final_output["baseline_raw_answer"] + "\n")
            f.write("=" * 90 + "\n")

        print(f"\n💾 Phase 4 최종 결과 저장 완료!")
        print(f"  👉 JSON 데이터: {self.output_json_path}")
        print(f"  👉 최종 비교 리포트 (필독): {self.output_txt_path}")


# === 실행부 ===
async def main():
    generator = TemporalSFTGenerator(
        shadow_dir="./shadow_workspace",
        llm_model="llama3"
    )
    final_output = await generator.run_generation_pipeline()
    generator.save_results(final_output)

    print("\n" + "=" * 90)
    print("🚀 [Proposed: Shadow TKG-RAG 시계열 추론 답변 미리보기]")
    print("=" * 90)
    print(final_output["proposed_temporal_answer"])
    print("=" * 90)

if __name__ == "__main__":
    asyncio.run(main())