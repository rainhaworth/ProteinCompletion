import itertools

from StructureEvaluation.atp_schedule_diagnostic import compare_schedule


def test_parallel_frontier_matches_packed_training_visibility_exhaustively():
    for length in range(1, 9):
        for size in range(1, length + 1):
            for motif in itertools.combinations(range(length), size):
                trace = compare_schedule(length, motif, "parallel")
                assert all(step["trained_pair"] for step in trace), (length, motif)
                assert all(not step["training_only"] and not step["inference_only"] for step in trace), (length, motif)


def test_serial_generation_can_see_a_different_context_than_training():
    parallel = compare_schedule(10, [3, 6], "parallel")
    serial = compare_schedule(10, [3, 6], "serial")

    assert all(not step["training_only"] and not step["inference_only"] for step in parallel)
    assert any(step["training_only"] or step["inference_only"] for step in serial)
    assert serial[1]["training_only"] == [4, 5, 7]
    assert serial[1]["target"] == 1
    assert any(not step["trained_pair"] for step in serial)


def test_parallel_direct_visibility_does_not_imply_same_multilayer_context():
    parallel = compare_schedule(10, [3, 6], "parallel")

    assert all(not step["training_only"] and not step["inference_only"] for step in parallel)
    assert any(step["contextual_row_mismatches"] for step in parallel)
