# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.chunk_size_utils import (
    compute_dynamic_initial_chunk_size,
)
from vllm_omni.model_executor.stage_input_processors.qwen3_tts import talker2code2wav_async_chunk

_FRAME = [1, 2, 3, 4]  # 4-codebook frame
_Q = len(_FRAME)  # num quantizers


def _req(rid: str, *, finished: bool, initial_codec_chunk_frames: int | None = None):
    ai = None
    if initial_codec_chunk_frames is not None:
        entry = SimpleNamespace(list_data=[initial_codec_chunk_frames])
        ai = SimpleNamespace(entries={"initial_codec_chunk_frames": entry})
    return SimpleNamespace(
        external_req_id=rid,
        is_finished=lambda: finished,
        additional_information=ai,
    )


def _tm(*, chunk_frames=25, left_context=25, initial_chunk=0, max_num_seqs=1):
    return SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        put_req_chunk=defaultdict(int),
        scheduler_max_num_seqs=max_num_seqs,
        connector=SimpleNamespace(
            config={
                "extra": {
                    "codec_chunk_frames": chunk_frames,
                    "codec_left_context_frames": left_context,
                    "initial_codec_chunk_frames": initial_chunk,
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


def test_does_not_emit_empty_chunk_when_not_finished():
    tm = _tm()
    payload = talker2code2wav_async_chunk(
        transfer_manager=tm,
        pooling_output={"audio_codes": torch.zeros((0,))},
        request=_req("rid-empty", finished=False),
    )
    assert payload is None


def test_flushes_tail_when_finished_without_pooler_output():
    tm = _tm()
    rid = "rid-tail"
    tm.code_prompt_token_ids[rid] = [_FRAME[:] for _ in range(24)]
    payload = talker2code2wav_async_chunk(
        transfer_manager=tm,
        pooling_output=None,
        request=_req(rid, finished=True),
        is_finished=True,
    )
    assert payload is not None
    assert payload["finished"].item() is True
    assert len(payload["code_predictor_codes"]) == _Q * 24


def test_emits_eof_marker_when_finished_with_no_frames():
    tm = _tm()
    payload = talker2code2wav_async_chunk(
        transfer_manager=tm,
        pooling_output=None,
        request=_req("rid-eof", finished=True),
        is_finished=True,
    )
    assert payload == {
        "code_predictor_codes": [],
        "finished": torch.tensor(True, dtype=torch.bool),
    }


# (chunk, lc, ic), n_frames, finished, expected (left_ctx, window_size) or None
_CASES = [
    # Normal path (ic=0): hold, then emit at boundary
    ((25, 25, 0), 24, False, None),
    ((25, 25, 0), 25, False, (0, 25)),
    # Initial phase: hold, then emit at IC boundary
    ((25, 25, 10), 9, False, None),
    ((25, 25, 10), 10, False, (0, 10)),
    # Normal phase after warmup: hold, then emit
    ((25, 25, 10), 30, False, None),
    ((25, 25, 10), 50, False, (25, 50)),
    # Finished flushes IC-phase tail and normal-phase tail
    ((25, 25, 10), 5, True, (0, 5)),
    ((25, 25, 10), 33, True, (25, 33)),
]


@pytest.mark.parametrize("config, n_frames, finished, expected", _CASES)
def test_streaming_phases(config, n_frames, finished, expected):
    chunk_frames, left_context, initial_chunk = config
    tm = _tm(chunk_frames=chunk_frames, left_context=left_context)
    req_ic = initial_chunk if initial_chunk > 0 else None
    payload = _call(tm, "r", n_frames=n_frames, finished=finished, req_ic=req_ic)

    if expected is None:
        assert payload is None
    else:
        exp_ctx, exp_window = expected
        assert payload is not None
        assert payload["left_context_size"] == exp_ctx
        assert len(payload["code_predictor_codes"]) == _Q * exp_window


def test_per_request_override_activates_initial_phase():
    tm = _tm(initial_chunk=0)
    payload = _call(tm, "r-override", n_frames=10, req_ic=10)
    assert payload is not None
    assert payload["left_context_size"] == 0
    assert len(payload["code_predictor_codes"]) == _Q * 10


def test_per_request_override_wins_over_stage_config():
    tm = _tm(initial_chunk=5)
    payload = _call(tm, "r-override2", n_frames=10, req_ic=15)
    assert payload is None


def test_per_request_override_bypasses_dynamic():
    tm = _tm(initial_chunk=10, max_num_seqs=4)
    payload = _call(tm, "r", n_frames=10, req_ic=10)
    assert payload is not None
    assert len(payload["code_predictor_codes"]) == _Q * 10


def test_dynamic_disabled_when_ic_zero():
    tm = _tm(initial_chunk=0, max_num_seqs=4)
    payload = _call(tm, "r", n_frames=10)
    assert payload is None


def test_dynamic_ic_adapts_mid_request():
    # First call: 1 active, IC=2, length=2, 2%2=0, emit
    tm = _tm(initial_chunk=10, max_num_seqs=4)
    p1 = _call(tm, "r", n_frames=2)
    assert p1 is not None
    assert len(p1["code_predictor_codes"]) == _Q * 2

    # Load increases, 4 active, IC=8, length=8, 8%8=0, emit
    for i in range(3):
        tm.code_prompt_token_ids[f"other-{i}"] = [[0]]
    p2 = _call(tm, "r", n_frames=8)
    assert p2 is not None
    assert len(p2["code_predictor_codes"]) == _Q * 8


@pytest.mark.parametrize(
    "active,max_bs,max_ic,expected",
    [
        (0, 4, 32, 2),  # zero load, min step
        (2, 4, 32, 8),  # mid load
        (4, 4, 32, 32),  # full load, cap at max_ic
        (10, 4, 16, 16),  # over capacity, still capped
        (0, 4, 1, 1),  # max_ic below min step
        (0, 0, 16, 2),  # zero max_batch_size edge case
    ],
)
def test_compute_dynamic_initial_chunk_size(active, max_bs, max_ic, expected):
    assert compute_dynamic_initial_chunk_size(active, max_bs, max_ic) == expected
