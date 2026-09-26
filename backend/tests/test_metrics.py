import pytest

from app.metrics import map_at_k, mrr_at_k, overlap_at_k, precision_at_k, recall_at_k


def test_recall_counts_a_hit_anywhere_in_top_k():
    rankings = [[9, 8, 1], [2, 7, 7], [5, 5, 5]]
    gold = [1, 2, 99]
    out = recall_at_k(rankings, gold, ks=(1, 3))
    assert out[1] == pytest.approx(1 / 3)
    assert out[3] == pytest.approx(2 / 3)


def test_recall_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        recall_at_k([[1]], [1, 2], ks=(1,))


def test_mrr_uses_reciprocal_of_first_hit_position():
    rankings = [[5, 1], [2, 9], [7, 7]]
    gold = [1, 2, 3]
    assert mrr_at_k(rankings, gold, k=2) == pytest.approx((0.5 + 1.0 + 0.0) / 3)


def test_mrr_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        mrr_at_k([[1]], [1, 2], k=1)


def test_mrr_ignores_hits_beyond_k():
    assert mrr_at_k([[9, 9, 1]], [1], k=2) == pytest.approx(0.0)


def test_precision_at_k_counts_label_overlap():
    ranked_labels = [[{"dog"}, {"cat"}, {"dog", "sofa"}, set()]]
    query_labels = [{"dog"}]
    assert precision_at_k(ranked_labels, query_labels, k=4) == pytest.approx(0.5)


def test_precision_at_k_with_no_query_labels_is_zero():
    assert precision_at_k([[{"dog"}]], [set()], k=1) == pytest.approx(0.0)


def test_map_at_k_weights_early_hits_more():
    early = map_at_k([[{"dog"}, set(), set()]], [{"dog"}], k=3)
    late = map_at_k([[set(), set(), {"dog"}]], [{"dog"}], k=3)
    assert early > late
    assert early == pytest.approx(1.0)
    assert late == pytest.approx(1 / 3)


def test_map_at_k_is_zero_when_nothing_relevant_retrieved():
    assert map_at_k([[set(), set()]], [{"dog"}], k=2) == pytest.approx(0.0)


def test_overlap_measures_agreement_between_two_rankings():
    a = [[1, 2, 3, 4]]
    b = [[3, 4, 5, 6]]
    assert overlap_at_k(a, b, k=4) == pytest.approx(0.5)


def test_overlap_is_one_for_identical_rankings():
    assert overlap_at_k([[1, 2]], [[2, 1]], k=2) == pytest.approx(1.0)
