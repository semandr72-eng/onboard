from services.rag_service import reciprocal_rank_fusion


def _hit(chunk_id: int) -> tuple[int, str, str]:
    return (chunk_id, f"Чанк {chunk_id}", f"Содержимое чанка {chunk_id}")


def test_shared_chunk_gets_highest_score():
    vector_hits = [_hit(1), _hit(2), _hit(3)]
    fts_hits = [_hit(3), _hit(1), _hit(4)]
    result = reciprocal_rank_fusion([vector_hits, fts_hits], top_n=4)
    # Чанк 1: 1/(60+1) + 1/(60+2); чанк 3: 1/(60+3) + 1/(60+1) — чанк 1 выше.
    assert result[0].chunk_id == 1
    assert result[1].chunk_id == 3
    assert result[0].score > result[1].score


def test_unique_chunk_keeps_its_source_rank_score():
    vector_hits = [_hit(1), _hit(2)]
    fts_hits = [_hit(2), _hit(3)]
    result = reciprocal_rank_fusion([vector_hits, fts_hits], top_n=3)
    by_id = {hit.chunk_id: hit.score for hit in result}
    assert by_id[1] == round(1.0 / 61, 6)
    assert by_id[3] == round(1.0 / 62, 6)


def test_top_n_limits_results():
    lists = [[_hit(i) for i in range(1, 6)]]
    result = reciprocal_rank_fusion(lists, top_n=2)
    assert len(result) == 2
    assert [hit.chunk_id for hit in result] == [1, 2]


def test_k_parameter_changes_scores():
    lists = [[_hit(1), _hit(2)]]
    flat = reciprocal_rank_fusion(lists, top_n=2, k=0)
    deep = reciprocal_rank_fusion(lists, top_n=2, k=100)
    assert flat[0].score > deep[0].score


def test_empty_lists_return_empty():
    assert reciprocal_rank_fusion([[], []], top_n=4) == []
