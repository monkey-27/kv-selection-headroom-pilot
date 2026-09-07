from kvheadroom.replay import ReplayLayout


def test_main_layout_alignment_and_protection():
    layout = ReplayLayout.main(prompt_tokens=23)
    empty = layout.keep_indices(0)
    full = layout.keep_indices((1 << 12) - 1)
    assert layout.prefix_reasoning_tokens == 895
    assert len(empty) == 23 + 127
    assert len(full) == 23 + 768 + 127
    assert empty[:23] == list(range(23))
    # reasoning 768..894 are the 127 cached recent tokens; token 895 is replayed as query.
    assert empty[23:] == list(range(23 + 768, 23 + 895))


def test_late_layout_partitions_every_old_token_once():
    layout = ReplayLayout.late(prompt_tokens=11, checkpoint=1501)
    flat = [i for block in layout.blocks for i in block]
    assert flat == list(range(1501 - 128))
    assert max(len(block) for block in layout.blocks) - min(len(block) for block in layout.blocks) <= 1
