"""
Evaluation harness: cross-product of speakers × listeners × cost values.

The harness is the main entry point for empirical evaluation.  Given a dataset
and lists of speaker/listener models, it runs all combinations and returns a
flat list of EvalRecord objects that can be written to JSONL or a DataFrame.

Design choices
--------------
- Speaker utterances are cached per (scene_id, speaker.name) so each speaker
  is called exactly once per scene, regardless of how many listeners or cost
  values are in the sweep.  This eliminates redundant LLM calls (the dominant
  cost driver) and ensures all listeners see identical utterances from a given
  speaker (no utterance-level variance confound).

- Listener posteriors are similarly cached per (scene_id, speaker.name,
  listener.name) so the same posterior is reused across cost sweep values.
  The CostAwareListener decision (ask vs. commit) depends only on the cost
  threshold, not on re-running the model.

- Speaker caching is parallelised across (scene, speaker) pairs; listener
  caching is parallelised across (scene, speaker, listener) triples.  Both
  phases use a ThreadPoolExecutor safe for I/O-bound LLM calls.

Cost impact (500 scenes, 2 VLM speakers, 1 VLM listener, 3 costs):
  Without caching : 6 000 speaker + 4 500 listener calls  ≈ $2.60
  With caching    : 1 000 speaker + 1 500 listener calls  ≈ $0.55
  (Llama-4-Maverick @ $0.18/M input + $0.72/M output, ~1 300 tokens/call)
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Sequence

from ..data.schema import EvalRecord, ListenerOutput, Scene, Utterance
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

    The function runs in three phases to minimise LLM calls:

      Phase 1 — Speaker cache
        Call speaker.speak(scene, target_idx) once per (scene, speaker) pair.
        Parallelised with n_workers threads.

      Phase 2 — Listener cache
        Call listener.listen(scene, utterance) once per (scene, speaker,
        listener) triple.  Parallelised with n_workers threads.

      Phase 3 — Decision sweep
        Apply the CostAwareListener threshold for each cost_c value using the
        cached posterior.  Pure Python, no LLM calls.

    Parameters
    ----------
    scenes      : evaluation scenes (test split recommended)
    speakers    : list of speaker models
    listeners   : list of base listener models (CostAwareListener wraps added
                  internally during the cost sweep)
    cost_values : clarification costs to sweep (e.g. [0.1, 0.25, 0.5])
    n_workers   : thread pool size for LLM calls (1 = sequential)
    verbose     : log phase-level progress

    Returns
    -------
    Flat list of EvalRecord, one per (scene × speaker × listener × cost_c).
    The list order is deterministic (scenes outer, speakers, listeners, costs).
    """

    # ── Phase 1: cache speaker utterances ────────────────────────────────────
    speaker_tasks = [(scene, speaker) for scene in scenes for speaker in speakers]
    utterance_cache: dict[tuple[str, str], Utterance] = {}

    if verbose:
        logger.info(
            f"Phase 1/3 — speaking: {len(speaker_tasks)} calls "
            f"({len(scenes)} scenes × {len(speakers)} speakers)"
        )

    def _speak(args):
        scene, speaker = args
        return (scene.id, speaker.name), speaker.speak(scene, scene.target_idx)

    _run_parallel(speaker_tasks, _speak, utterance_cache, n_workers, verbose,
                  total=len(speaker_tasks), label="speaker")

    # ── Phase 2: cache listener posteriors ───────────────────────────────────
    listener_tasks = [
        (scene, speaker, listener)
        for scene in scenes
        for speaker in speakers
        for listener in listeners
    ]
    posterior_cache: dict[tuple[str, str, str], ListenerOutput] = {}

    if verbose:
        logger.info(
            f"Phase 2/3 — listening: {len(listener_tasks)} calls "
            f"({len(scenes)} scenes × {len(speakers)} speakers × {len(listeners)} listeners)"
        )

    def _listen(args):
        scene, speaker, listener = args
        utt = utterance_cache[(scene.id, speaker.name)]
        return (scene.id, speaker.name, listener.name), listener.listen(scene, utt)

    _run_parallel(listener_tasks, _listen, posterior_cache, n_workers, verbose,
                  total=len(listener_tasks), label="listener")

    # ── Phase 3: cost sweep (no LLM calls) ───────────────────────────────────
    if verbose:
        n_decisions = len(scenes) * len(speakers) * len(listeners) * len(cost_values)
        logger.info(f"Phase 3/3 — cost sweep: {n_decisions} decisions (no LLM calls)")

    records: list[EvalRecord] = []
    for scene in scenes:
        for speaker in speakers:
            utt = utterance_cache.get((scene.id, speaker.name))
            if utt is None:
                continue
            for listener in listeners:
                l_out = posterior_cache.get((scene.id, speaker.name, listener.name))
                if l_out is None:
                    continue
                for cost_c in cost_values:
                    record = _apply_cost_decision(scene, speaker, listener, utt, l_out, cost_c)
                    records.append(record)

    if verbose:
        logger.info(f"Grid complete — {len(records)} records collected")

    return records


# ── Parallelism helper ────────────────────────────────────────────────────────

def _run_parallel(tasks, fn, cache: dict, n_workers: int, verbose: bool,
                  total: int, label: str) -> None:
    """Submit tasks to thread pool, store (key, value) results into cache."""
    errors = 0
    if n_workers > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = {pool.submit(fn, t): t for t in tasks}
            done = 0
            for fut in as_completed(futs):
                done += 1
                try:
                    key, val = fut.result()
                    cache[key] = val
                except Exception as e:
                    errors += 1
                    logger.error(f"{label} task failed: {e}")
                if verbose and done % max(1, total // 10) == 0:
                    logger.info(f"  {label}: {done}/{total} done")
    else:
        for i, task in enumerate(tasks):
            try:
                key, val = fn(task)
                cache[key] = val
            except Exception as e:
                errors += 1
                logger.error(f"{label} task {i} failed: {e}")
            if verbose and (i + 1) % max(1, total // 10) == 0:
                logger.info(f"  {label}: {i+1}/{total} done")
    if errors:
        rate = errors / total
        msg  = f"{label} phase: {errors}/{total} tasks failed ({rate:.1%})"
        if rate > 0.05:
            logger.error(f"HIGH FAILURE RATE — {msg}. Results may be biased. Re-run with --workers 1.")
        else:
            logger.warning(msg)


# ── Cost-threshold decision ───────────────────────────────────────────────────

def _apply_cost_decision(
    scene:    Scene,
    speaker:  BaseSpeaker,
    listener: BaseListener,
    utt:      Utterance,
    l_out:    ListenerOutput,
    cost_c:   float,
) -> EvalRecord:
    """Apply EU threshold to a cached ListenerOutput and produce an EvalRecord."""
    ca = CostAwareListener(base_listener=listener, cost_c=cost_c)
    decision = ca._decide(scene, l_out)

    action        = decision.action
    predicted_idx = l_out.predicted_idx
    correct       = predicted_idx == scene.target_idx

    ann = scene.entropy_annotation
    meta = l_out.listener_meta or {}
    return EvalRecord(
        scene_id=scene.id,
        speaker_type=speaker.name,
        listener_type=listener.name,
        cost_c=cost_c,
        utterance=utt.text,
        action=action,
        predicted_idx=predicted_idx,
        target_idx=scene.target_idx,
        correct=correct,
        eu_commit=decision.eu_commit,
        eu_ask=decision.eu_ask,
        entropy=referential_entropy(l_out.posterior),
        brier_score=brier_score(l_out.posterior, scene.target_idx),
        min_desc_length=ann.min_desc_length if ann else None,
        ambiguity_tier=ann.ambiguity_tier if ann else None,
        pred_x=meta.get("pred_x"),
        pred_y=meta.get("pred_y"),
    )


# ── Serialization helpers ─────────────────────────────────────────────────────

def records_to_dicts(records: list[EvalRecord]) -> list[dict]:
    return [asdict(r) for r in records]


def records_to_dataframe(records: list[EvalRecord]):
    """Convert to pandas DataFrame (pandas must be installed)."""
    import pandas as pd
    return pd.DataFrame(records_to_dicts(records))
