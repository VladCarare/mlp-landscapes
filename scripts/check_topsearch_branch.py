"""Check that the topsearch branch required by the current stage is installed.

Both topsearch submodules install as the distribution ``topsearch``, so only
one of them is importable at a time and ``RUNME.sh`` swaps between them part
way through. Running a stage against the wrong branch fails a long way from
the cause, for example::

    ValueError: not enough values to unpack (expected 3, got 2)

from ``combine_results.py`` when the ``mlp_run`` branch is still installed.
This script turns that into an immediate, actionable error.

Usage::

    python scripts/check_topsearch_branch.py {mlp_run|analysis}
"""

from __future__ import annotations

import sys

import networkx as nx

# The two branches model the kinetic transition network differently, and that
# difference is what breaks the analysis scripts: only a MultiGraph can hold
# the parallel edges needed when two minima are joined by several transition
# states.
GRAPH_TYPES: dict[str, type] = {
    "mlp_run": nx.Graph,
    "analysis": nx.MultiGraph,
}


def installed_branch() -> tuple[str, str]:
    """Return the name and location of the topsearch branch that is importable.

    Raises
    ------
    SystemExit
        If topsearch cannot be imported, or its network type matches neither
        known branch.
    """
    try:
        import topsearch
        from topsearch.data.kinetic_transition_network import (
            KineticTransitionNetwork,
        )
    except ImportError as exc:
        raise SystemExit(
            f"ERROR: could not import topsearch ({exc}).\n"
            "Install it with: pip install -e external/topsearch-mlp_run/"
        ) from exc

    graph_type = type(KineticTransitionNetwork().G)
    for branch, expected in GRAPH_TYPES.items():
        if graph_type is expected:
            return branch, topsearch.__file__

    raise SystemExit(
        f"ERROR: unrecognised topsearch at {topsearch.__file__}: "
        f"KineticTransitionNetwork.G is {graph_type.__name__}, expected one of "
        + ", ".join(t.__name__ for t in GRAPH_TYPES.values())
    )


def main(expected: str) -> None:
    """Exit non-zero unless ``expected`` is the installed topsearch branch."""
    if expected not in GRAPH_TYPES:
        raise SystemExit(
            f"ERROR: unknown branch {expected!r}, "
            f"expected one of {', '.join(GRAPH_TYPES)}"
        )

    branch, location = installed_branch()
    if branch != expected:
        raise SystemExit(
            f"ERROR: this stage needs the topsearch '{expected}' branch, but "
            f"the '{branch}' branch is installed, from\n    {location}\n"
            f"Install the right one with:\n"
            f"    pip install -e external/topsearch-{expected}/"
        )

    print(f"topsearch '{branch}' branch is installed, from {location}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
