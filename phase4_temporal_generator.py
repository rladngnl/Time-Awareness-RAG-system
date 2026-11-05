import os
import json
import asyncio
from typing import Dict, Any
import ollama

class TemporalSFTGenerator:
    def __init__(
        self,
        shadow_dir: str = "./shadow_workspace",
        llm_model: str = "llama3"
    ):
        self.shadow_dir = shadow_dir
        self.llm_model = llm_model
        
        self.input_json_path = os.path.join(shadow_dir, "phase3_joined_context_v2.json")
        self.output_json_path = os.path.join(shadow_dir, "phase4_final_output_v2.json")
        self.output_txt_path = os.path.join(shadow_dir, "phase4_final_answer_report_v2.txt")
        
        self.client = ollama.AsyncClient(host="http://localhost:11434")

    def load_phase3_context(self) -> Dict[str, Any]:
        with open(self.input_json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # 💡 이전 에러의 원인: 여기에 raw_context를 인자로 완벽히 추가하고 아래에서 제대로 넘겨줍니다.
    async def generate_temporal_answer(self, query: str, cypher_table: str, raw_context: str) -> str:
        system_prompt = """You are an advanced Temporal Knowledge Graph Reasoning AI.
You are provided with two inputs:
1. 'Chronological Cypher Table': A strictly time-sorted table of graph events. Use this as your PRIMARY anchor.
2. 'Supporting Raw Context': Detailed source descriptions to prevent entity confusion.

Follow these steps:
1. Trace the timeline strictly following the Chronological Cypher Table from top to bottom.
2. If an entity's state changes multiple times (e.g., CEO changes), explicitly identify the latest entry.
3. Answer all parts of the user's question accurately."""

        user_prompt = f"""### 1. PRIMARY ANCHOR: Chronological Cypher Table
{cypher_table}

### 2. SUPPORTING EVIDENCE: Retrieved Context
{raw_context}

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
            options={"temperature": 0.1}
        )
        return response['message']['content']

    async def generate_baseline_answer(self, query: str, raw_context: str) -> str:
        system_prompt = "You are a helpful AI assistant. Answer the user's question based on the provided context."
        user_prompt = f"### Retrieved Context\n{raw_context}\n\n### User Question\n{query}\n\nAnswer:"
        
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

        print("\n⏳ 1. [Proposed] 하이브리드 컨텍스트 기반 시계열 추론 생성 중...")
        temporal_answer = await self.generate_temporal_answer(query, cypher_table, raw_context)

        print("⏳ 2. [Baseline] 일반 LightRAG Raw Context 기반 생성 중...")
        baseline_answer = await self.generate_baseline_answer(query, raw_context)

        return {
            "query": query,
            "chronological_cypher_table": cypher_table,
            "proposed_temporal_answer": temporal_answer,
            "baseline_raw_answer": baseline_answer
        }

    def save_results(self, final_output: Dict[str, Any]):
        with open(self.output_json_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, ensure_ascii=False, indent=4)

        with open(self.output_txt_path, "w", encoding="utf-8") as f:
            f.write("=" * 90 + "\n🏆 [Phase 4 최종 리포트 v2.0] Shadow TKG-RAG 결과\n" + "=" * 90 + "\n\n")
            f.write(f"❓ Query: {final_output['query']}\n\n")
            f.write("🚀 [Proposed] Shadow TKG-RAG 최종 답변\n" + "-" * 90 + "\n")
            f.write(final_output["proposed_temporal_answer"] + "\n\n")
            f.write("🐢 [Baseline] 순정 LightRAG 컨텍스트 기반 답변\n" + "-" * 90 + "\n")
            f.write(final_output["baseline_raw_answer"] + "\n")

async def main():
    generator = TemporalSFTGenerator()
    final_output = await generator.run_generation_pipeline()
    generator.save_results(final_output)

if __name__ == "__main__":
    asyncio.run(main())