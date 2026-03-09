# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.chunk_size_utils import (
    compute_dynamic_initial_chunk_size,
    max_ic_for_chunk_size,
)
from vllm_omni.model_executor.stage_input_processors.qwen3_tts import talker2code2wav_async_chunk

_FRAME = [1, 2, 3, 4]
_Q = len(_FRAME)


def _req(rid, *, finished, initial_codec_chunk_frames=None):
    ai = None
    if initial_codec_chunk_frames is not None:
        entry = SimpleNamespace(list_data=[initial_codec_chunk_frames])
        ai = SimpleNamespace(entries={"initial_codec_chunk_frames": entry})
    return SimpleNamespace(
        external_req_id=rid,
        is_finished=lambda: finished,
        additional_information=ai,
    )


def _tm(*, chunk_frames=25, left_context=25, max_num_seqs=1):
    return SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        scheduler_max_num_seqs=max_num_seqs,
        connector=SimpleNamespace(
            config={
                "extra": {
                    "codec_chunk_frames": chunk_frames,
                    "codec_left_context_frames": left_context,
                }
            }
        ),
    )


def _call(tm, rid, *, n_frames, finished=False, req_ic=None):
    tm.code_prompt_token_ids[rid] = [_FRAME[:] for _ in range(n_frames)]
    return talker2code2wav_async_chunk(
        transfer_manager=tm,
        pooling_output={"audio_codes": torch.zeros((0,))},
        request=_req(rid, finished=finished, initial_codec_chunk_frames=req_ic),
        is_finished=finished,
    )


def test_empty_returns_none():
    tm = _tm()
    p = talker2code2wav_async_chunk(
        transfer_manager=tm,
        pooling_output={"audio_codes": torch.zeros((0,))},
        request=_req("r", finished=False),
    )
    assert p is None


def test_eof_marker_when_finished_empty():
    tm = _tm()
    p = talker2code2wav_async_chunk(
        transfer_manager=tm,
        pooling_output=None,
        request=_req("r", finished=True),
        is_finished=True,
    )
    assert p == {"code_predictor_codes": [], "finished": torch.tensor(True, dtype=torch.bool)}


def test_flush_on_finish():
    tm = _tm()
    tm.code_prompt_token_ids["r"] = [_FRAME[:] for _ in range(24)]
    p = talker2code2wav_async_chunk(
        transfer_manager=tm,
        pooling_output=None,
        request=_req("r", finished=True),
        is_finished=True,
    )
    assert p is not None
    assert p["finished"].item() is True
    assert len(p["code_predictor_codes"]) == _Q * 24


_CASES = [
    # Dynamic IC=16, cs=25, initial_coverage=16
    ((25, 25, 0), 24, False, None),  # IC phase: 24%16!=0 -> hold
    ((25, 25, 0), 25, False, None),  # transition: adjusted=9, hold (no replay)
    ((25, 25, 0), 41, False, (16, 41)),  # first normal emit, lc=16
    # Per-request IC=10, cs=25, initial_coverage=20
    ((25, 25, 10), 9, False, None),  # IC: hold
    ((25, 25, 10), 10, False, (0, 10)),  # IC: emit at boundary
    ((25, 25, 10), 25, False, None),  # transition: hold (no replay)
    ((25, 25, 10), 45, False, (20, 45)),  # first normal emit, lc=20
    ((25, 25, 10), 5, True, (0, 5)),  # finished flushes IC tail
    ((25, 25, 10), 33, True, (20, 33)),  # finished flushes normal tail
    # IC=8, cs=16: IC divides chunk_size evenly (edge case)
    ((16, 25, 8), 8, False, (0, 8)),  # IC: emit
    ((16, 25, 8), 16, False, None),  # transition: hold (no replay)
    ((16, 25, 8), 24, False, (8, 24)),  # first normal emit, lc=8
    # Per-request override: IC=15 at n_frames=10 -> 10%15!=0 -> hold
    ((25, 25, 15), 10, False, None),
]


@pytest.mark.parametrize("config, n_frames, finished, expected", _CASES)
def test_streaming_phases(config, n_frames, finished, expected):
    chunk_frames, left_context, req_ic_val = config
    tm = _tm(chunk_frames=chunk_frames, left_context=left_context)
    req_ic = req_ic_val if req_ic_val > 0 else None
    payload = _call(tm, "r", n_frames=n_frames, finished=finished, req_ic=req_ic)

    if expected is None:
        assert payload is None
    else:
        exp_ctx, exp_window = expected
        assert payload is not None
        assert payload["left_context_size"] == exp_ctx
        assert len(payload["code_predictor_codes"]) == _Q * exp_window


def test_dynamic_ic_adapts_to_load():
    # chunk_size=25 -> max_ic=16, steps=[2,4,8,16]
    tm = _tm(max_num_seqs=8)

    # Low load (1/8) -> IC=2 -> emit at 2
    p1 = _call(tm, "r", n_frames=2)
    assert p1 is not None
    assert len(p1["code_predictor_codes"]) == _Q * 2

    # High load: add 4 others -> active=5/8 -> IC=8 -> emit at 8
    for i in range(4):
        tm.code_prompt_token_ids[f"other-{i}"] = [[0]]
    p2 = _call(tm, "r", n_frames=8)
    assert p2 is not None
    assert len(p2["code_predictor_codes"]) == _Q * 8

    # Requests past initial phase still count in load factor
    tm2 = _tm(max_num_seqs=4)
    for i in range(3):
        tm2.code_prompt_token_ids[f"long-{i}"] = [[0]] * 50  # well past cs=25
    # active=4/4=1.0 -> IC=16
    p3 = _call(tm2, "new", n_frames=16)
    assert p3 is not None
    assert len(p3["code_predictor_codes"]) == _Q * 16


def test_ic_load_change_mid_request():
    """IC stateless: load spike mid-request shifts initial_coverage."""
    tm = _tm(chunk_frames=25, left_context=25, max_num_seqs=8)

    # Low load -> IC=2 -> emit at frame 2
    p1 = _call(tm, "r", n_frames=2)
    assert p1 is not None

    # Spike load: 6 others -> IC=16 -> initial_coverage=16
    for i in range(6):
        tm.code_prompt_token_ids[f"other-{i}"] = [[0]] * 10

    # adjusted=25-16=9, 9%25!=0 -> hold
    assert _call(tm, "r", n_frames=25) is None

    # First normal emit at 16+25=41
    p3 = _call(tm, "r", n_frames=41)
    assert p3 is not None
    assert p3["left_context_size"] == 16


@pytest.mark.parametrize(
    "active,max_bs,max_ic,expected",
    [
        (0, 4, 32, 2),  # zero load -> min step
        (2, 4, 32, 8),  # mid load
        (4, 4, 32, 32),  # full load
        (10, 4, 16, 16),  # over capacity, capped
        (0, 4, 1, 1),  # max_ic below min step
        (0, 0, 16, 2),  # zero capacity edge case
    ],
)
def test_compute_dynamic_initial_chunk_size(active, max_bs, max_ic, expected):
    assert compute_dynamic_initial_chunk_size(active, max_bs, max_ic) == expected


@pytest.mark.parametrize(
    "chunk_size,expected",
    [
        (25, 16),
        (50, 32),
        (70, 64),
        (8, 4),
        (4, 2),
        (2, 1),
        (1, 1),
    ],
)
def test_max_ic_for_chunk_size(chunk_size, expected):
    assert max_ic_for_chunk_size(chunk_size) == expected
