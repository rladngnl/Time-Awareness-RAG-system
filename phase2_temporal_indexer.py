import os
import json
import asyncio
from typing import List, Dict, Any
import ollama

class TemporalEdgeIndexer:
    """
    Phase 2: Phase 1에서 추출된 [Edge + 원문 텍스트] 데이터를 읽어,
    Ollama를 통해 각 관계의 발생/유효 시간(start_time, end_time)을 추출하고
    별도의 Shadow Temporal Index 파일로 저장합니다.
    """
    def __init__(
        self, 
        workspace_dir: str = "./shadow_workspace",
        llm_model: str = "llama3"
    ):
        self.workspace_dir = workspace_dir
        self.llm_model = llm_model
        
        self.input_json_path = os.path.join(workspace_dir, "phase1_mapped_edges.json")
        self.output_json_path = os.path.join(workspace_dir, "temporal_shadow_index.json")
        self.output_txt_path = os.path.join(workspace_dir, "phase2_temporal_report.txt")
        
        self.client = ollama.AsyncClient(host="http://localhost:11434")

    async def extract_time_for_edge(self, edge: Dict[str, Any]) -> Dict[str, str]:
        """단일 엣지와 원문 문맥을 LLM에 전달하여 시작/종료 시점을 JSON으로 추출합니다."""
        
        system_prompt = """You are a precise temporal information extraction system for a Knowledge Graph.
Analyze the given Graph Edge (Source, Target, Relation Description) and its Source Text.
Extract the exact time period (start_time and end_time) when this specific relation occurred or was valid.

Rules:
1. Format dates as YYYY or YYYY-MM (e.g., "2018", "2020-05").
2. If the event is a single point-in-time action (e.g., founded, acquired, released), set both start_time and end_time to that same date, or set end_time to "Present" if the state continues.
3. If a clear start and end exist (e.g., started in 2018 and completed in Dec 2019), extract both accurately.
4. If no time information can be inferred for this relation, use "Unknown".
5. Respond ONLY with a valid JSON object containing keys: "start_time", "end_time", and "temporal_note" (a brief 1-sentence reason)."""

        user_prompt = f"""[Graph Edge Info]
- Source Entity: {edge['source']}
- Target Entity: {edge['target']}
- Relation Keywords: {edge['keywords']}
- Relation Description: {edge['relation_desc']}

[Source Text]
"{edge['source_text']}"

Extract the temporal metadata in JSON format:"""

        try:
            response = await self.client.chat(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                format="json",  # 💡 파싱 에러 방지를 위한 JSON 출력 강제
                options={"temperature": 0.0}  # 일관된 사실 추출을 위해 온도를 0으로 설정
            )
            
            result_text = response['message']['content']
            parsed_time = json.loads(result_text)
            
            return {
                "start_time": str(parsed_time.get("start_time", "Unknown")),
                "end_time": str(parsed_time.get("end_time", "Unknown")),
                "temporal_note": str(parsed_time.get("temporal_note", ""))
            }
        except Exception as e:
            print(f"⚠️ [{edge['edge_id']}] 시간 추출 중 오류 발생: {e}")
            return {
                "start_time": "Unknown",
                "end_time": "Unknown",
                "temporal_note": f"Extraction failed: {str(e)}"
            }

    async def build_temporal_index(self) -> Dict[str, Any]:
        if not os.path.exists(self.input_json_path):
            raise FileNotFoundError(f"❌ Phase 1 결과 파일을 찾을 수 없습니다: {self.input_json_path}")

        with open(self.input_json_path, "r", encoding="utf-8") as f:
            mapped_edges: List[Dict[str, Any]] = json.load(f)

        print(f"📂 Phase 1 데이터 로드 완료: 총 {len(mapped_edges)}개의 엣지에 대해 시간축 추출을 시작합니다...")

        temporal_index = {}

        for idx, edge in enumerate(mapped_edges, 1):
            src = edge["source"]
            tgt = edge["target"]
            print(f"⏳ [{idx}/{len(mapped_edges)}] 시간 추출 중: ({src}) -> ({tgt}) ...")
            
            time_meta = await self.extract_time_for_edge(edge)
            
            # Phase 3에서 방향에 상관없이 (A, B) 노드로 빠르게 Join하기 위한 정렬 키 생성
            sorted_nodes = sorted([src, tgt])
            lookup_key = f"{sorted_nodes[0]}__{sorted_nodes[1]}"

            # 기존 엣지 정보에 시간 메타데이터 결합
            temporal_edge_record = {
                "edge_id": edge["edge_id"],
                "lookup_key": lookup_key,
                "source": src,
                "target": tgt,
                "keywords": edge["keywords"],
                "relation_desc": edge["relation_desc"],
                "start_time": time_meta["start_time"],
                "end_time": time_meta["end_time"],
                "temporal_note": time_meta["temporal_note"],
                "chunk_ids": edge["chunk_ids"]
            }
            
            temporal_index[lookup_key] = temporal_edge_record
            print(f"   ✅ 추출 완료: [{time_meta['start_time']} ~ {time_meta['end_time']}] ({time_meta['temporal_note']})")

        return temporal_index

    def save_temporal_index(self, temporal_index: Dict[str, Any]):
        """구축된 시간축 인덱스를 JSON과 리포트용 TXT 파일로 저장합니다."""
        
        # 1. JSON 저장 (Phase 3에서 Join할 DB 파일)
        with open(self.output_json_path, "w", encoding="utf-8") as f:
            json.dump(temporal_index, f, ensure_ascii=False, indent=4)

        # 2. TXT 리포트 저장 (Chronological Cypher Table 미리보기 포함)
        # start_time 기준으로 오름차순 정렬 (Unknown은 맨 뒤로)
        sorted_records = sorted(
            temporal_index.values(),
            key=lambda x: (x["start_time"] == "Unknown", x["start_time"])
        )

        with open(self.output_txt_path, "w", encoding="utf-8") as f:
            f.write("=" * 85 + "\n")
            f.write(f"🕒 [Phase 2 결과 리포트] Shadow Temporal Index 구축 완료 (총 {len(sorted_records)}개 엣지)\n")
            f.write("=" * 85 + "\n\n")

            f.write("### 1. Chronological Cypher Table 미리보기 (시간순 정렬)\n")
            f.write("| Timestamp | Cypher Relation Pattern | Description |\n")
            f.write("| :--- | :--- | :--- |\n")
            for rec in sorted_records:
                time_range = rec["start_time"] if rec["start_time"] == rec["end_time"] else f"{rec['start_time']} ~ {rec['end_time']}"
                cypher_pattern = f"({rec['source']})-[:{rec['keywords']}]->({rec['target']})"
                f.write(f"| {time_range} | {cypher_pattern} | {rec['relation_desc']} |\n")
            
            f.write("\n" + "=" * 85 + "\n")
            f.write("### 2. 개별 엣지 상세 추출 내역\n\n")
            for idx, rec in enumerate(sorted_records, 1):
                f.write(f"[{idx}] Lookup Key: {rec['lookup_key']}\n")
                f.write(f"  • 관계: ({rec['source']}) ---[{rec['keywords']}]---> ({rec['target']})\n")
                f.write(f"  • 추출 시간: {rec['start_time']} ~ {rec['end_time']}\n")
                f.write(f"  • 추론 근거: {rec['temporal_note']}\n")
                f.write(f"  • 설명: {rec['relation_desc']}\n")
                f.write("-" * 85 + "\n")

        print(f"\n💾 Shadow Temporal Index 저장 완료!")
        print(f"  👉 시간축 DB (Phase 3 연동용): {self.output_json_path}")
        print(f"  👉 시간축 리포트 (확인용): {self.output_txt_path}")

# === 실행부 ===
async def main():
    indexer = TemporalEdgeIndexer(
        workspace_dir="./shadow_workspace",
        llm_model="llama3"  # 본인 로컬 Ollama 모델명과 일치 확인
    )
    temporal_index = await indexer.build_temporal_index()
    indexer.save_temporal_index(temporal_index)

if __name__ == "__main__":
    asyncio.run(main())