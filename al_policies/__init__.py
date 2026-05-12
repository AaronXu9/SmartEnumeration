"""Active-learning MEL-budget allocation policies.

Drop-in replacements for the existing `allocate_budget()` rule in
[run_srg_batch.py](../run_srg_batch.py). All implement the
`AllocationPolicy` Protocol from `base.py`. See [docs/AL_Benchmark.md](../docs/AL_Benchmark.md)
for the design, and [al_benchmark/run_benchmark.py](../al_benchmark/run_benchmark.py)
for the offline comparison harness.
"""
from al_policies.base import (
    AllocationPolicy,
    DictHistory,
    EmptyHistory,
    HistoryView,
    POLICY_REGISTRY,
    get,
    register,
)

# Importing the policy modules registers them in POLICY_REGISTRY as a side
# effect. Keep this list in sync with the modules below — the benchmark
# harness and the live runner both rely on it.
from al_policies import baseline  # noqa: F401,E402
from al_policies import greedy  # noqa: F401,E402
from al_policies import bandit  # noqa: F401,E402

# ML policy is gated by optional scikit-learn at import time.
try:
    from al_policies import ml  # noqa: F401,E402
    _ML_AVAILABLE = True
except ImportError:
    _ML_AVAILABLE = False

__all__ = [
    "AllocationPolicy",
    "DictHistory",
    "EmptyHistory",
    "HistoryView",
    "POLICY_REGISTRY",
    "get",
    "register",
]
