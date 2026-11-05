import os
import json
import networkx as nx
from typing import List, Dict, Any

class ShadowGraphParser:
    """
    Phase 1: LightRAG의 워크스페이스에서 정적 그래프(.graphml)와 청크(.json)를 읽어
    [Edge + 원본 텍스트] 쌍을 추출하고, 이를 별도의 파일(JSON, TXT)로 저장합니다.
    """
    def __init__(
        self, 
        lightrag_dir: str = "./lightrag_workspace",
        output_dir: str = "./shadow_workspace"
    ):
        self.lightrag_dir = lightrag_dir
        self.output_dir = output_dir
        
        self.graphml_path = os.path.join(lightrag_dir, "graph_chunk_entity_relation.graphml")
        self.chunks_json_path = os.path.join(lightrag_dir, "kv_store_text_chunks.json")
        
        # 래퍼 전용 저장 폴더 생성
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def extract_edges_with_context(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.graphml_path):
            raise FileNotFoundError(f"❌ 그래프 파일을 찾을 수 없습니다: {self.graphml_path}")
        if not os.path.exists(self.chunks_json_path):
            raise FileNotFoundError(f"❌ 청크 JSON 파일을 찾을 수 없습니다: {self.chunks_json_path}")

        print("📂 워크스페이스에서 그래프 및 원본 청크 데이터 로드 중...")
        G = nx.read_graphml(self.graphml_path)
        
        with open(self.chunks_json_path, "r", encoding="utf-8") as f:
            text_chunks_db = json.load(f)

        print(f"✅ 로드 완료: 총 노드 {G.number_of_nodes()}개, 총 엣지 {G.number_of_edges()}개 발견")

        mapped_edges = []

        for u, v, edge_data in G.edges(data=True):
            source_node = str(u).strip('"')
            target_node = str(v).strip('"')
            description = edge_data.get("description", "").strip('"')
            keywords = edge_data.get("keywords", "").strip('"')
            
            raw_source_ids = edge_data.get("source_id", "").strip('"')
            chunk_ids = raw_source_ids.split("<SEP>") if raw_source_ids else []

            source_texts = []
            for cid in chunk_ids:
                if cid in text_chunks_db:
                    chunk_content = text_chunks_db[cid].get("content", "").strip()
                    source_texts.append(chunk_content)
                else:
                    source_texts.append(f"[원문 찾을 수 없음: {cid}]")

            combined_source_text = "\n".join(source_texts)

            edge_info = {
                "edge_id": f"{source_node}__{target_node}",
                "source": source_node,
                "target": target_node,
                "relation_desc": description,
                "keywords": keywords,
                "chunk_ids": chunk_ids,
                "source_text": combined_source_text
            }
            mapped_edges.append(edge_info)

        return mapped_edges

    def save_results(self, mapped_edges: List[Dict[str, Any]]):
        """추출된 결과를 JSON(기계용)과 TXT(사람 확인용) 두 가지 파일로 저장합니다."""
        json_output_path = os.path.join(self.output_dir, "phase1_mapped_edges.json")
        txt_output_path = os.path.join(self.output_dir, "phase1_mapped_edges_report.txt")

        # 1. JSON 파일 저장 (Phase 2 입력용)
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(mapped_edges, f, ensure_ascii=False, indent=4)
            
        # 2. TXT 리포트 파일 저장 (가독성 확인용)
        with open(txt_output_path, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write(f"🔍 [Phase 1 결과 리포트] 총 {len(mapped_edges)}개의 엣지 매핑 완료\n")
            f.write("=" * 70 + "\n\n")
            
            for idx, item in enumerate(mapped_edges, 1):
                f.write(f"[{idx}] Edge ID: {item['edge_id']}\n")
                f.write(f"  • 구조: ({item['source']}) ---[{item['keywords']}]---> ({item['target']})\n")
                f.write(f"  • 관계 설명: {item['relation_desc']}\n")
                f.write(f"  • 출처 청크 ID: {', '.join(item['chunk_ids'])}\n")
                f.write(f"  • 매핑된 원문:\n    \"{item['source_text']}\"\n")
                f.write("-" * 70 + "\n\n")

        print(f"\n💾 결과 파일 저장 완료!")
        print(f"  👉 JSON 데이터 (Phase 2 연동용): {json_output_path}")
        print(f"  👉 TXT 리포트 (내용 확인용): {txt_output_path}")

# === 실행부 ===
if __name__ == "__main__":
    parser = ShadowGraphParser()
    edges_data = parser.extract_edges_with_context()
    parser.save_results(edges_data)