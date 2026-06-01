# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_omni.model_executor.models.fish_speech.configuration_fish_speech import (
    FishSpeechFastARConfig,
    FishSpeechSlowARConfig,
)
from vllm_omni.model_executor.models.fish_speech.fish_speech_fast_ar import FishSpeechFastAR
from vllm_omni.model_executor.models.fish_speech.fish_speech_slow_ar import (
    FishSpeechSlowARForConditionalGeneration,
)


def _make_sampling_fast_ar(monkeypatch, fixed_logits: torch.Tensor) -> FishSpeechFastAR:
    """Build a Fast AR whose transformer is stubbed to emit ``fixed_logits`` (only the sampling tail runs for real)."""
    fast_config = FishSpeechFastARConfig(
        vocab_size=fixed_logits.shape[-1],
        num_codebooks=4,
        dim=8,
        n_head=2,
        n_local_heads=1,
        head_dim=4,
        n_layer=1,
        intermediate_size=16,
        max_seq_len=8,
    )
    slow_config = FishSpeechSlowARConfig(
        vocab_size=32,
        dim=8,
        n_head=2,
        n_local_heads=1,
        head_dim=4,
        n_layer=1,
        intermediate_size=16,
        codebook_size=fixed_logits.shape[-1],
        num_codebooks=4,
        semantic_begin_id=4,
        semantic_end_id=4 + fixed_logits.shape[-1] - 1,
    )
    fast_ar = object.__new__(FishSpeechFastAR)
    nn.Module.__init__(fast_ar)
    fast_ar.config = fast_config
    fast_ar.slow_ar_config = slow_config
    fast_ar.fast_project_in = nn.Identity()
    fast_ar.fast_embeddings = nn.Embedding(fast_config.vocab_size, fast_config.hidden_size)
    fast_ar.fast_output = nn.Identity()
    fast_ar.fast_norm = nn.Identity()
    fast_ar._num_codebooks = fast_config.num_codebooks
    fast_ar._fast_dim = fast_config.hidden_size
    fast_ar._embed_buf = None
    fast_ar._pos_ids = None
    fast_ar._k_cache = None
    fast_ar._v_cache = None
    fast_ar._compiled_model_fwd = None
    fast_ar._compile_attempted = True
    fast_ar._compile_failed = False
    fast_ar._disable_compile_for_graph = False

    bsz = 1

    def fake_run_model_one(step_input, step_pos_ids, cache_pos):
        # fast_output / fast_norm are Identity, so returning fixed_logits == the model output.
        return fixed_logits.expand(step_input.shape[0], -1).clone()

    monkeypatch.setattr(fast_ar, "_run_model_one", fake_run_model_one)
    return fast_ar, slow_config, bsz


def test_fast_ar_sampling_is_deterministic_with_seed(monkeypatch):
    torch.manual_seed(0)
    fixed_logits = torch.randn(1, 64)
    fast_ar, slow_config, _ = _make_sampling_fast_ar(monkeypatch, fixed_logits)

    hidden = torch.zeros(1, slow_config.hidden_size, dtype=torch.float32)
    semantic = torch.full((1,), slow_config.semantic_begin_id, dtype=torch.long)

    kw = dict(do_sample=True, temperature=0.8, top_k=30, top_p=0.9)
    c1 = fast_ar(hidden, semantic, seed=123, **kw)
    c2 = fast_ar(hidden, semantic, seed=123, **kw)
    c3 = fast_ar(hidden, semantic, seed=999, **kw)

    assert torch.equal(c1, c2), "same seed must give identical residual codes"
    # Code 0 is the deterministic semantic code, independent of sampling.
    assert int(c1[0, 0]) == int(semantic[0]) - slow_config.semantic_begin_id
    # With different seeds the residual codes (steps >= 1) should differ in practice.
    assert not torch.equal(c1[:, 1:], c3[:, 1:])


def test_fast_ar_greedy_is_seed_independent(monkeypatch):
    torch.manual_seed(0)
    fixed_logits = torch.randn(1, 64)
    fast_ar, slow_config, _ = _make_sampling_fast_ar(monkeypatch, fixed_logits)

    hidden = torch.zeros(1, slow_config.hidden_size, dtype=torch.float32)
    semantic = torch.full((1,), slow_config.semantic_begin_id, dtype=torch.long)

    g1 = fast_ar(hidden, semantic, do_sample=False)
    g2 = fast_ar(hidden, semantic, do_sample=False, seed=5)
    assert torch.equal(g1, g2), "greedy decode must be independent of seed"


def test_fast_ar_exponential_sampler_matches_multinomial_distribution(monkeypatch):
    """The exponential-max draw must be distribution-equivalent to multinomial (guards the swap)."""
    torch.manual_seed(0)
    vocab = 64
    fixed_logits = torch.randn(1, vocab)
    top_k, top_p, temperature = 30, 0.9, 0.8
    inv_t = 1.0 / temperature

    # Reproduce the exact post-filter probability vector from the sampler.
    scaled = fixed_logits * inv_t
    topk_vals, _ = scaled.topk(min(top_k, vocab), dim=-1)
    scaled = scaled.masked_fill(scaled < topk_vals[:, -1:], float("-inf"))
    sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    sorted_logits[(cumulative - sorted_probs) >= top_p] = float("-inf")
    scaled = sorted_logits.scatter(1, sorted_indices, sorted_logits)
    probs = F.softmax(scaled, dim=-1)

    n = 200_000
    big = probs.expand(n, -1).contiguous()

    g_mult = torch.Generator().manual_seed(7)
    mult = torch.multinomial(big, 1, replacement=True, generator=g_mult).reshape(-1)
    hist_mult = torch.bincount(mult, minlength=vocab).float() / n

    g_exp = torch.Generator().manual_seed(7)
    q = torch.empty_like(big)
    q.exponential_(generator=g_exp)
    expd = big.clone().div_(q).argmax(dim=-1)
    hist_exp = torch.bincount(expd, minlength=vocab).float() / n

    # Both must track the true probabilities within Monte-Carlo error.
    assert torch.allclose(hist_mult, probs.reshape(-1), atol=0.01)
    assert torch.allclose(hist_exp, probs.reshape(-1), atol=0.01)
    # And track each other.
    assert (hist_mult - hist_exp).abs().max().item() < 0.01


def test_fast_ar_exponential_sampler_never_selects_filtered_token():
    """Filtered tokens (probs == 0) must never be drawn (no 0/0 -> NaN -> spurious argmax pick)."""
    torch.manual_seed(1)
    vocab = 32
    probs = torch.zeros(1, vocab)
    allowed = torch.tensor([3, 7, 11, 20])
    probs[0, allowed] = torch.tensor([0.4, 0.3, 0.2, 0.1])

    n = 500_000
    big = probs.expand(n, -1).contiguous().clone()
    q = torch.empty_like(big)
    q.exponential_()
    drawn = big.div_(q).argmax(dim=-1)
    assert torch.isin(drawn, allowed).all(), "sampler selected a zero-probability token"


def test_fast_ar_reuses_dense_position_id_buffer(monkeypatch):
    fast_config = FishSpeechFastARConfig(
        vocab_size=16,
        num_codebooks=4,
        dim=8,
        n_head=2,
        n_local_heads=1,
        head_dim=4,
        n_layer=1,
        intermediate_size=16,
        max_seq_len=5,
    )
    slow_config = FishSpeechSlowARConfig(
        vocab_size=32,
        dim=8,
        n_head=2,
        n_local_heads=1,
        head_dim=4,
        n_layer=1,
        intermediate_size=16,
        codebook_size=16,
        num_codebooks=4,
        semantic_begin_id=4,
        semantic_end_id=19,
    )
    fast_ar = object.__new__(FishSpeechFastAR)
    nn.Module.__init__(fast_ar)
    fast_ar.config = fast_config
    fast_ar.slow_ar_config = slow_config
    fast_ar.fast_project_in = nn.Identity()
    fast_ar.fast_embeddings = nn.Embedding(fast_config.vocab_size, fast_config.hidden_size)
    fast_ar.fast_output = nn.Linear(fast_config.hidden_size, fast_config.vocab_size, bias=False)
    fast_ar.fast_norm = nn.Identity()
    fast_ar._num_codebooks = fast_config.num_codebooks
    fast_ar._fast_dim = fast_config.hidden_size
    fast_ar._embed_buf = None
    fast_ar._pos_ids = None
    fast_ar._k_cache = None
    fast_ar._v_cache = None
    fast_ar._compiled_model_fwd = None
    fast_ar._compile_attempted = True
    fast_ar._compile_failed = False
    fast_ar._disable_compile_for_graph = False

    seen_position_ids: list[torch.Tensor] = []

    def fake_run_model_one(step_input, step_pos_ids, cache_pos):
        seen_position_ids.append(step_pos_ids)
        return torch.zeros(
            step_input.shape[0],
            fast_config.hidden_size,
            dtype=step_input.dtype,
            device=step_input.device,
        )

    monkeypatch.setattr(fast_ar, "_run_model_one", fake_run_model_one)

    hidden = torch.zeros(3, slow_config.hidden_size, dtype=torch.float32)
    semantic = torch.full((3,), slow_config.semantic_begin_id, dtype=torch.long)
    fast_ar(hidden, semantic, do_sample=False)

    assert fast_ar._pos_ids is not None
    assert fast_ar._pos_ids.shape == (3, fast_config.num_codebooks + 1)
    assert len(seen_position_ids) == fast_config.num_codebooks
    for step, step_pos_ids in enumerate(seen_position_ids):
        assert step_pos_ids.shape == (3,)
        assert step_pos_ids.untyped_storage().data_ptr() == fast_ar._pos_ids.untyped_storage().data_ptr()
        assert step_pos_ids.stride() == (fast_ar._pos_ids.stride(0),)
        expected = torch.full((3,), step)
        assert torch.equal(step_pos_ids.cpu(), expected)


def test_talker_mtp_does_not_mutate_input():
    model = object.__new__(FishSpeechSlowARForConditionalGeneration)
    nn.Module.__init__(model)
    model._semantic_begin_id = 4
    model._semantic_end_id = 19
    model._codebook_size = 16
    model._num_codebooks = 4
    model.codebook_embeddings = nn.Embedding(model._num_codebooks * model._codebook_size, 8)
    model.fast_ar = lambda **_: torch.tensor(
        [
            [0, 1, 2, 3],
            [15, 14, 13, 12],
        ],
        dtype=torch.long,
    )
    input_ids = torch.tensor([4, 2], dtype=torch.long)
    input_embeds = torch.randn(2, 8)
    input_embeds_before = input_embeds.clone()
    last_hidden = torch.randn(2, 8, dtype=torch.bfloat16)
    text_step = torch.zeros(2, 8, dtype=torch.bfloat16)

    out1, _ = FishSpeechSlowARForConditionalGeneration.talker_mtp(
        model,
        input_ids,
        input_embeds,
        last_hidden,
        text_step,
    )

    assert torch.equal(input_embeds, input_embeds_before)
    assert out1.untyped_storage().data_ptr() != input_embeds.untyped_storage().data_ptr()


def test_talker_mtp_forwards_sampling_params():
    model = object.__new__(FishSpeechSlowARForConditionalGeneration)
    nn.Module.__init__(model)
    model._semantic_begin_id = 4
    model._semantic_end_id = 19
    model._codebook_size = 16
    model._num_codebooks = 4
    model.codebook_embeddings = nn.Embedding(model._num_codebooks * model._codebook_size, 8)
    seen = {}

    def fake_fast_ar(**kwargs):
        seen.update(kwargs)
        return torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

    model.fast_ar = fake_fast_ar
    input_ids = torch.tensor([4], dtype=torch.long)
    input_embeds = torch.randn(1, 8)
    last_hidden = torch.randn(1, 8, dtype=torch.bfloat16)
    text_step = torch.zeros(1, 8, dtype=torch.bfloat16)

    FishSpeechSlowARForConditionalGeneration.talker_mtp(
        model,
        input_ids,
        input_embeds,
        last_hidden,
        text_step,
        do_sample=False,
        temperature=0.1,
        top_k=5,
        top_p=0.7,
    )

    assert seen["do_sample"] is False
    assert seen["temperature"] == 0.1
    assert seen["top_k"] == 5
    assert seen["top_p"] == 0.7
