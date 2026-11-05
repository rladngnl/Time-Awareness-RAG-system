import os
import shutil
import asyncio

# 우리가 만든 각 모듈과 Phase 클래스 임포트
from test_module1 import OllamaLightRAGWrapper
from phase1_graph_parser import ShadowGraphParser
from phase2_temporal_indexer import TemporalEdgeIndexer
from phase3_subgraph_joiner import TemporalSubgraphJoiner
from phase4_temporal_generator import TemporalSFTGenerator

async def run_trap_experiment():
    lightrag_dir = "./lightrag_workspace"
    shadow_dir = "./shadow_workspace"
    trap_data_path = "./data/sample_complex_history.txt"

    print("=" * 90)
    print("🧪 [시계열 함정 벤치마크] 기존 워크스페이스 초기화 및 클린 테스트 시작")
    print("=" * 90)

    # 1. 완벽한 클린 테스트를 위해 기존 워크스페이스 폴더 삭제
    if os.path.exists(lightrag_dir):
        shutil.rmtree(lightrag_dir)
        print(f"🧹 기존 {lightrag_dir} 삭제 완료")
    if os.path.exists(shadow_dir):
        shutil.rmtree(shadow_dir)
        print(f"🧹 기존 {shadow_dir} 삭제 완료")

    # ---------------------------------------------------------
    # Step 1: Module 1 (LightRAG 정적 인덱싱)
    # ---------------------------------------------------------
    print("\n" + "-" * 90)
    print("🚀 [Step 1] Module 1: 함정 데이터셋(sample_complex_history.txt) 인덱싱")
    print("-" * 90)
    module1 = OllamaLightRAGWrapper(working_dir=lightrag_dir)
    await module1.initialize()
    await module1.ingest_document(trap_data_path)

    # ---------------------------------------------------------
    # Step 2: Phase 1 (GraphML 엣지 및 원본 청크 매핑)
    # ---------------------------------------------------------
    print("\n" + "-" * 90)
    print("🚀 [Step 2] Phase 1: 정적 그래프 엣지 및 원문 매핑 추출")
    print("-" * 90)
    parser = ShadowGraphParser(lightrag_dir=lightrag_dir, output_dir=shadow_dir)
    edges_data = parser.extract_edges_with_context()
    parser.save_results(edges_data)

    # ---------------------------------------------------------
    # Step 3: Phase 2 (Ollama 기반 병렬 시간축 DB 구축)
    # ---------------------------------------------------------
    print("\n" + "-" * 90)
    print("🚀 [Step 3] Phase 2: 엣지별 유효 기간 추출 및 Shadow Temporal Index 구축")
    print("-" * 90)
    indexer = TemporalEdgeIndexer(workspace_dir=shadow_dir, llm_model="llama3")
    temporal_index = await indexer.build_temporal_index()
    indexer.save_temporal_index(temporal_index)

    # ---------------------------------------------------------
    # Step 4: Phase 3 (복합 시계열 질문 투입 및 Cypher Table 생성)
    # ---------------------------------------------------------
    print("\n" + "-" * 90)
    print("🚀 [Step 4] Phase 3: 시계열 서브그래프 가로채기 및 Chronological Cypher Table 생성")
    print("-" * 90)
    
    # 💡 3가지 함정(소유권 변천사, 현재 최종 소유주, CEO 교체 순서 및 현재 CEO)을 모두 묻는 고난도 질문
    complex_trap_query = (
        "Trace the complete chronological ownership history of Project Chronos from its creation to the present, "
        "identify who currently owns Project Chronos, and explain the exact chronological order of NovaTech's CEOs "
        "including who the current CEO is."
    )
    
    joiner = TemporalSubgraphJoiner(lightrag_dir=lightrag_dir, shadow_dir=shadow_dir)
    await joiner.initialize()
    joined_payload = await joiner.intercept_and_join(complex_trap_query, mode="hybrid")
    joiner.save_results(joined_payload)

    # ---------------------------------------------------------
    # Step 5: Phase 4 (Shadow TKG-RAG vs Baseline 답변 비교 생성)
    # ---------------------------------------------------------
    print("\n" + "-" * 90)
    print("🚀 [Step 5] Phase 4: 최종 시계열 추론 답변(Proposed) vs 순정 LightRAG(Baseline) 비교")
    print("-" * 90)
    generator = TemporalSFTGenerator(shadow_dir=shadow_dir, llm_model="llama3")
    final_output = await generator.run_generation_pipeline()
    generator.save_results(final_output)

    print("\n" + "=" * 90)
    print("🎉 모든 벤치마크 파이프라인 실행 완료!")
    print("👉 './shadow_workspace/phase4_final_answer_report.txt' 파일을 열어 두 모델의 추론 차이를 확인하세요!")
    print("=" * 90)

if __name__ == "__main__":
    asyncio.run(run_trap_experiment())