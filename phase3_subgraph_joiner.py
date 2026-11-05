import os
import json
import asyncio
from typing import Dict, Any, List
import numpy as np
import ollama
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

async def custom_ollama_llm(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    client = ollama.AsyncClient(host="http://localhost:11434")
    messages = []
    if system_prompt: messages.append({"role": "system", "content": system_prompt})
    if history_messages: messages.extend(history_messages)
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

class TemporalSubgraphJoiner:
    def __init__(self, lightrag_dir="./lightrag_workspace", shadow_dir="./shadow_workspace"):
        self.lightrag_dir = lightrag_dir
        self.shadow_dir = shadow_dir
        
        self.temporal_index_path = os.path.join(shadow_dir, "temporal_shadow_index_v2.json")
        self.output_json_path = os.path.join(shadow_dir, "phase3_joined_context_v2.json")
        self.output_txt_path = os.path.join(shadow_dir, "phase3_cypher_table_report_v2.txt")

        self.rag = LightRAG(
            working_dir=self.lightrag_dir,
            llm_model_func=custom_ollama_llm,
            embedding_func=EmbeddingFunc(embedding_dim=768, max_token_size=8192, func=custom_ollama_embed)
        )

    async def initialize(self):
        await self.rag.initialize_storages()

    def load_temporal_index(self) -> Dict[str, Any]:
        with open(self.temporal_index_path, "r", encoding="utf-8") as f:
            return json.load(f)

    async def intercept_and_join(self, query: str, mode: str = "hybrid") -> Dict[str, Any]:
        param = QueryParam(mode=mode, only_need_context=True)
        raw_context = await self.rag.aquery(query, param=param)
        
        temporal_index = self.load_temporal_index()
        raw_context_lower = raw_context.lower() if raw_context else ""
        matched_events = []

        for lookup_key, edge_record in temporal_index.items():
            src, tgt = edge_record["source"], edge_record["target"]
            if src.lower() in raw_context_lower and tgt.lower() in raw_context_lower:
                for ev in edge_record.get("temporal_events", []):
                    ev_copy = ev.copy()
                    ev_copy["source"] = src
                    ev_copy["target"] = tgt
                    matched_events.append(ev_copy)

        sorted_events = sorted(matched_events, key=lambda x: (x["start_time"] == "Unknown", x["start_time"]))
        
        lines = ["| Timestamp | Cypher Relation Pattern | Exact Factual Event Detail |", "| :--- | :--- | :--- |"]
        for ev in sorted_events:
            time_str = ev["start_time"] if ev["start_time"] == ev["end_time"] else f"{ev['start_time']} ~ {ev['end_time']}"
            lines.append(f"| {time_str} | ({ev['source']})-[:{ev['action_label']}]->({ev['target']}) | {ev['event_detail']} |")

        cypher_table = "\n".join(lines) if sorted_events else "No temporal graph edges matched."

        return {
            "query": query,
            "retrieval_mode": mode,
            "matched_edge_count": len(sorted_events),
            "chronological_cypher_table": cypher_table,
            "sorted_temporal_edges": sorted_events,
            "raw_lightrag_context": raw_context
        }

    def save_results(self, payload: Dict[str, Any]):
        with open(self.output_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=4)
        with open(self.output_txt_path, "w", encoding="utf-8") as f:
            f.write(payload["chronological_cypher_table"])

async def main():
    joiner = TemporalSubgraphJoiner()
    await joiner.initialize()
    result = await joiner.intercept_and_join("test")
    joiner.save_results(result)

if __name__ == "__main__":
    asyncio.run(main())