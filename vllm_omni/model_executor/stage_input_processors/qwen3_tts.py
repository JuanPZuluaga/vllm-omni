"""Stage input processor for Qwen3-TTS: Talker -> Code2Wav."""

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.model_executor.stage_input_processors.chunk_size_utils import (
    compute_dynamic_initial_chunk_size,
)

logger = init_logger(__name__)


def talker2code2wav(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
) -> list[Any]:
    """Non-async processor: wait for talker to finish, then pass all codes to code2wav at once."""
    from vllm_omni.inputs.data import OmniTokensPrompt
    from vllm_omni.model_executor.stage_input_processors.qwen3_omni import _validate_stage_inputs

    talker_outputs = _validate_stage_inputs(stage_list, engine_input_source)
    code2wav_inputs: list[OmniTokensPrompt] = []
    for talker_output in talker_outputs:
        output = talker_output.outputs[0]
        # audio_codes shape: [num_frames, Q] where Q=num_quantizers (16)
        audio_codes = output.multimodal_output["audio_codes"].to(torch.long)
        # Filter zero-padded frames (EOS/invalid steps), matching _extract_last_frame behavior
        valid_mask = audio_codes.any(dim=1)
        audio_codes = audio_codes[valid_mask]
        # Code2Wav expects codebook-major flat: [Q*num_frames]
        codec_codes = audio_codes.transpose(0, 1).cpu().reshape(-1).tolist()
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=codec_codes,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return code2wav_inputs


def _extract_last_frame(pooling_output: dict[str, Any]) -> torch.Tensor | None:
    audio_codes = pooling_output.get("audio_codes")
    if not isinstance(audio_codes, torch.Tensor) or audio_codes.numel() == 0:
        return None
    if audio_codes.ndim == 2:
        frame = audio_codes[-1]
        if frame.numel() == 0 or not bool(frame.any().item()):
            return None
        return frame.to(torch.long).reshape(-1)
    if audio_codes.ndim == 1:
        return audio_codes.to(torch.long).reshape(-1)
    raise ValueError(f"Invalid audio_codes shape for Qwen3-TTS async_chunk: {tuple(audio_codes.shape)}")


def talker2code2wav_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> dict[str, Any] | None:
    request_id = request.external_req_id
    finished = bool(is_finished or request.is_finished())

    if isinstance(pooling_output, dict):
        frame = _extract_last_frame(pooling_output)
        if frame is not None:
            codec_codes = frame.cpu().tolist()
            transfer_manager.code_prompt_token_ids[request_id].append(codec_codes)
    elif not finished:
        # Some steps may not produce pooling_output. Only flush on finish.
        return None

    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size = int(cfg.get("codec_chunk_frames", 25))
    left_context_size_config = int(cfg.get("codec_left_context_frames", 25))
    initial_chunk_size = int(cfg.get("initial_codec_chunk_frames", 0))
    # Per-request override (takes priority over stage config and dynamic)
    per_request_override = False
    additional_information = getattr(request, "additional_information", None)
    if (
        additional_information is not None
        and hasattr(additional_information, "entries")
        and "initial_codec_chunk_frames" in additional_information.entries
    ):
        entry = additional_information.entries["initial_codec_chunk_frames"]
        if entry.list_data is not None and len(entry.list_data) == 1:
            initial_chunk_size = int(entry.list_data[0])
            per_request_override = True
    # Dynamic IC: recomputed every call so IC adapts to current load mid-request.
    if not per_request_override and 0 < initial_chunk_size < chunk_size:
        active = sum(1 for v in transfer_manager.code_prompt_token_ids.values() if v)
        capacity = getattr(transfer_manager, "scheduler_max_num_seqs", 1)
        initial_chunk_size = compute_dynamic_initial_chunk_size(active, capacity, initial_chunk_size)
        logger.debug(
            "Dynamic IC: active=%d, capacity=%d, ic=%d, req=%s", active, capacity, initial_chunk_size, request_id
        )
    if chunk_size <= 0 or left_context_size_config < 0 or initial_chunk_size < 0:
        raise ValueError(
            f"Invalid codec chunk config: codec_chunk_frames={chunk_size}, "
            f"codec_left_context_frames={left_context_size_config}, "
            f"initial_codec_chunk_frames={initial_chunk_size}"
        )
    if initial_chunk_size > chunk_size:
        logger.warning(
            "initial_codec_chunk_frames=%d > codec_chunk_frames=%d, clamping to codec_chunk_frames.",
            initial_chunk_size,
            chunk_size,
        )
        initial_chunk_size = chunk_size
    length = len(transfer_manager.code_prompt_token_ids[request_id])

    if length <= 0:
        if finished:
            return {
                "code_predictor_codes": [],
                "finished": torch.tensor(True, dtype=torch.bool),
            }
        return None

    # Initial phase (length < chunk_size): emit at IC boundaries with smaller chunks.
    # Normal phase (length >= chunk_size): standard chunk_size cadence.
    # Both use the same sliding window — no per-request state needed.
    in_initial_phase = initial_chunk_size > 0 and initial_chunk_size < chunk_size and length < chunk_size
    effective_chunk = initial_chunk_size if in_initial_phase else chunk_size

    if not finished and length % effective_chunk != 0:
        return None

    # On finish, flush whatever remains.
    context_length = length % effective_chunk if (finished and length % effective_chunk != 0) else effective_chunk
    end_index = min(length, left_context_size_config + context_length)
    left_context_size = max(0, end_index - context_length)
    window_frames = transfer_manager.code_prompt_token_ids[request_id][-end_index:]

    code_predictor_codes = torch.tensor(window_frames).transpose(0, 1).reshape(-1).tolist()

    return {
        "code_predictor_codes": code_predictor_codes,
        "left_context_size": left_context_size,
        "finished": torch.tensor(finished, dtype=torch.bool),
    }
