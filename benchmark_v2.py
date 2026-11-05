import asyncio
import os

# v2.0으로 업데이트된 클래스들 임포트
from phase2_temporal_indexer import TemporalEdgeIndexer
from phase3_subgraph_joiner import TemporalSubgraphJoiner
from phase4_temporal_generator import TemporalSFTGenerator

async def run_v2_pipeline():
    lightrag_dir = "./lightrag_workspace"
    shadow_dir = "./shadow_workspace"
    
    print("=" * 90)
    print("🚀 [Shadow TKG-RAG v2.0] 고도화 파이프라인 실행 시작")
    print("=" * 90)

    # ---------------------------------------------------------
    # 1. Phase 2 (v2.0): 1:N 다중 시점 이벤트 분해 추출
    # ---------------------------------------------------------
    print("\n▶ [Phase 2] 다중 시점 이벤트 분해(1:N Decomposition) 진행 중...")
    indexer = TemporalEdgeIndexer(workspace_dir=shadow_dir, llm_model="llama3")
    temporal_index = await indexer.build_temporal_index()
    indexer.save_temporal_index(temporal_index)

    # ---------------------------------------------------------
    # 2. Phase 3 (v2.0): 하위 이벤트 평탄화(Flatten) 및 Cypher Table 생성
    # ---------------------------------------------------------
    print("\n▶ [Phase 3] 서브그래프 가로채기 및 v2.0 Chronological Cypher Table 생성 중...")
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
    # 3. Phase 4 (v2.0): 하이브리드 컨텍스트(Table + Raw) 주입 추론
    # ---------------------------------------------------------
    print("\n▶ [Phase 4] Hybrid Context(Cypher Table + Raw Context) 기반 최종 추론 중...")
    generator = TemporalSFTGenerator(shadow_dir=shadow_dir, llm_model="llama3")
    final_output = await generator.run_generation_pipeline()
    generator.save_results(final_output)

    print("\n" + "=" * 90)
    print("🎉 v2.0 파이프라인 실행 완료!")
    print("👉 './shadow_workspace/phase4_final_answer_report.txt'를 열어 극복된 결과를 확인하세요.")
    print("=" * 90)

if __name__ == "__main__":
    asyncio.run(run_v2_pipeline())