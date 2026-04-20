"""
Evaluation harness: cross-product of speakers × listeners × cost values.

The harness is the main entry point for empirical evaluation.  Given a dataset
and lists of speaker/listener models, it runs all combinations and returns a
flat list of EvalRecord objects that can be written to JSONL or a DataFrame.

Design choices
--------------
- Each (speaker, listener, cost_c) triple is evaluated independently so results
  are comparable across the full grid.
- Speakers are called once per scene and their utterances are cached so that all
  listeners receive the same utterance from each speaker.  This ensures that
  listener comparison is not confounded by utterance variance.
- Progress is reported via tqdm (if installed) so long runs are observable.
- `run_grid` supports optional parallelism via `n_workers` (thread-based,
  safe for I/O-bound LLM calls).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Sequence

from ..data.schema import EvalRecord, Scene, Utterance
from ..speakers.base import BaseSpeaker
from ..listeners.base import BaseListener
from ..listeners.cost_aware import CostAwareListener
from ..metrics.core import brier_score, referential_entropy

logger = logging.getLogger(__name__)


def run_grid(
    scenes:      list[Scene],
    speakers:    list[BaseSpeaker],
    listeners:   list[BaseListener],
    cost_values: list[float] = (0.1, 0.25, 0.5),
    n_workers:   int = 1,
    verbose:     bool = True,
) -> list[EvalRecord]:
    """
    Evaluate all (speaker, listener, cost_c) triples on all scenes.

    Parameters
    ----------
    scenes      : evaluation scenes (test split recommended)
    speakers    : list of speaker models
    listeners   : list of base listener models (CostAwareListener wraps are added internally)
    cost_values : clarification costs to sweep
    n_workers   : thread pool size for LLM calls (1 = sequential)
    verbose     : log progress

    Returns
    -------
    Flat list of EvalRecord, one per (scene × speaker × listener × cost_c).
    """
    records: list[EvalRecord] = []

    # Build the full grid of tasks
    tasks = [
        (scene, speaker, listener, c)
        for scene in scenes
        for speaker in speakers
        for listener in listeners
        for c in cost_values
    ]

    if verbose:
        logger.info(f"Running {len(tasks)} evaluations "
                    f"({len(scenes)} scenes × {len(speakers)} speakers × "
                    f"{len(listeners)} listeners × {len(cost_values)} cost values)")

    def _run_one(args):
        scene, speaker, listener, cost_c = args
        return _evaluate_one(scene, speaker, listener, cost_c)

    if n_workers > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = {pool.submit(_run_one, t): t for t in tasks}
            done = 0
            for fut in as_completed(futs):
                done += 1
                try:
                    record = fut.result()
                    records.append(record)
                except Exception as e:
                    task = futs[fut]
                    logger.error(f"Task failed ({task[1].name}/{task[2].name}): {e}")
                if verbose and done % max(1, len(tasks) // 20) == 0:
                    logger.info(f"  {done}/{len(tasks)} done")
    else:
        for i, task in enumerate(tasks):
            try:
                records.append(_run_one(task))
            except Exception as e:
                logger.error(f"Task {i} failed: {e}")
            if verbose and (i + 1) % max(1, len(tasks) // 20) == 0:
                logger.info(f"  {i+1}/{len(tasks)} done")

    return records


# ── Single-scene evaluation ───────────────────────────────────────────────────

def _evaluate_one(
    scene:    Scene,
    speaker:  BaseSpeaker,
    listener: BaseListener,
    cost_c:   float,
) -> EvalRecord:
    """Run one (scene, speaker, listener, cost_c) combination."""
    # Speaker produces utterance
    utterance: Utterance = speaker.speak(scene, scene.target_idx)

    # Wrap listener with cost-aware policy
    ca_listener = CostAwareListener(base_listener=listener, cost_c=cost_c)
    l_out = ca_listener.listen(scene, utterance)
    decision = l_out.listener_meta.get("clarification")

    action        = decision.action if decision else "commit"
    predicted_idx = l_out.predicted_idx
    correct       = predicted_idx == scene.target_idx

    ann = scene.entropy_annotation
    return EvalRecord(
        scene_id=scene.id,
        speaker_type=speaker.name,
        listener_type=listener.name,
        cost_c=cost_c,
        utterance=utterance.text,
        action=action,
        predicted_idx=predicted_idx,
        target_idx=scene.target_idx,
        correct=correct,
        eu_commit=decision.eu_commit if decision else max(l_out.posterior),
        eu_ask=decision.eu_ask if decision else 1.0 - cost_c,
        entropy=referential_entropy(l_out.posterior),
        brier_score=brier_score(l_out.posterior, scene.target_idx),
        min_desc_length=ann.min_desc_length if ann else None,
        ambiguity_tier=ann.ambiguity_tier if ann else None,
    )


# ── Serialization helpers ─────────────────────────────────────────────────────

def records_to_dicts(records: list[EvalRecord]) -> list[dict]:
    return [asdict(r) for r in records]


def records_to_dataframe(records: list[EvalRecord]):
    """Convert to pandas DataFrame (pandas must be installed)."""
    import pandas as pd
    return pd.DataFrame(records_to_dicts(records))
