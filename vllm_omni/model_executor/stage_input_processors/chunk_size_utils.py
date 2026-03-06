# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

_IC_STEPS = (2, 4, 8, 16, 32)


def compute_dynamic_initial_chunk_size(
    active_requests: int,
    max_batch_size: int,
    max_ic: int,
) -> int:
    """Select IC from power-of-2 steps based on load factor."""
    steps = [s for s in _IC_STEPS if s <= max_ic]
    if not steps or max_batch_size <= 0:
        return min(_IC_STEPS[0], max_ic) if max_ic > 0 else _IC_STEPS[0]
    load_factor = min(active_requests / max_batch_size, 1.0)
    idx = int(round(load_factor * (len(steps) - 1)))
    return steps[idx]
