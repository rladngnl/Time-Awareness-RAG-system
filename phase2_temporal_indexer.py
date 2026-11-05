import os
import json
import asyncio
from typing import List, Dict, Any
import ollama

class TemporalEdgeIndexer:
    def __init__(
        self, 
        workspace_dir: str = "./shadow_workspace",
        llm_model: str = "llama3"
    ):
        self.workspace_dir = workspace_dir
        self.llm_model = llm_model
        
        self.input_json_path = os.path.join(workspace_dir, "phase1_mapped_edges.json")
        self.output_json_path = os.path.join(workspace_dir, "temporal_shadow_index_v2.json")
        self.output_txt_path = os.path.join(workspace_dir, "phase2_temporal_report_v2.txt")
        
        self.client = ollama.AsyncClient(host="http://localhost:11434")

    async def extract_time_for_edge(self, edge: Dict[str, Any]) -> List[Dict[str, str]]:
        system_prompt = """You are an expert Temporal Knowledge Graph extractor.
Analyze the given Graph Edge (Source, Target) and the Source Text.
Extract ALL distinct temporal events/states between the Source and Target mentioned in the Source Text.
Respond ONLY with a valid JSON object containing an "events" list:
{
  "events": [
    {
      "start_time": "YYYY-MM",
      "end_time": "YYYY-MM or Present",
      "action_label": "APPOINTED_CEO / DISMISSED / SOLD_PROJECT / ACQUIRED",
      "event_detail": "Exact factual description"
    }
  ]
}"""
        user_prompt = f"""[Graph Edge Info]
- Entity 1: {edge['source']}
- Entity 2: {edge['target']}
- Description: {edge['relation_desc']}

[Original Source Text]
"{edge['source_text']}"

Extract all distinct temporal events in JSON format:"""

        try:
            response = await self.client.chat(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                format="json",
                options={"temperature": 0.0}
            )
            parsed_data = json.loads(response['message']['content'])
            events = parsed_data.get("events", [])
            
            if not events or not isinstance(events, list):
                return [{"start_time": "Unknown", "end_time": "Unknown", "action_label": edge["keywords"].split(",")[0].strip().upper().replace(" ", "_"), "event_detail": edge["relation_desc"]}]
            
            cleaned_events = []
            for ev in events:
                cleaned_events.append({
                    "start_time": str(ev.get("start_time", "Unknown")),
                    "end_time": str(ev.get("end_time", "Unknown")),
                    "action_label": str(ev.get("action_label", "RELATED_TO")).upper().replace(" ", "_"),
                    "event_detail": str(ev.get("event_detail", edge["relation_desc"]))
                })
            return cleaned_events
        except Exception as e:
            print(f"⚠️ [{edge['edge_id']}] 시간 추출 오류: {e}")
            return [{"start_time": "Unknown", "end_time": "Unknown", "action_label": "ERROR", "event_detail": edge["relation_desc"]}]

    async def build_temporal_index(self) -> Dict[str, Any]:
        with open(self.input_json_path, "r", encoding="utf-8") as f:
            mapped_edges = json.load(f)

        print(f"📂 Phase 1 데이터 로드 완료: 총 {len(mapped_edges)}개 엣지 다중 시점 분해 중...")
        temporal_index = {}
        for idx, edge in enumerate(mapped_edges, 1):
            src, tgt = edge["source"], edge["target"]
            print(f"⏳ [{idx}/{len(mapped_edges)}] 시계열 분해: ({src}) <-> ({tgt})")
            
            events_list = await self.extract_time_for_edge(edge)
            sorted_nodes = sorted([src, tgt])
            lookup_key = f"{sorted_nodes[0]}__{sorted_nodes[1]}"

            temporal_index[lookup_key] = {
                "edge_id": edge["edge_id"],
                "source": src,
                "target": tgt,
                "temporal_events": events_list,
                "chunk_ids": edge["chunk_ids"]
            }
        return temporal_index

    def save_temporal_index(self, temporal_index: Dict[str, Any]):
        with open(self.output_json_path, "w", encoding="utf-8") as f:
            json.dump(temporal_index, f, ensure_ascii=False, indent=4)

        flattened_events = []
        for rec in temporal_index.values():
            for ev in rec["temporal_events"]:
                ev_copy = ev.copy()
                ev_copy["source"] = rec["source"]
                ev_copy["target"] = rec["target"]
                flattened_events.append(ev_copy)

        sorted_events = sorted(flattened_events, key=lambda x: (x["start_time"] == "Unknown", x["start_time"]))

        with open(self.output_txt_path, "w", encoding="utf-8") as f:
            f.write("=" * 95 + "\n")
            f.write(f"🕒 [Phase 2 v2.0 리포트] 다중 시점 분해 완료\n")
            f.write("=" * 95 + "\n\n")
            for ev in sorted_events:
                f.write(f"| {ev['start_time']}~{ev['end_time']} | ({ev['source']})-[:{ev['action_label']}]->({ev['target']}) | {ev['event_detail']} |\n")

async def main():
    indexer = TemporalEdgeIndexer()
    temporal_index = await indexer.build_temporal_index()
    indexer.save_temporal_index(temporal_index)

if __name__ == "__main__":
    asyncio.run(main())