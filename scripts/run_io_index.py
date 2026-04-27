"""
Run io-index listener on all 6 datasets, re-using cached speaker utterances
from results/vllm_honest/, then merge the new records into those files.

io-index outputs a single integer (no distribution) so max(posterior)=1.0
always — it never asks and CPA == Accuracy for all cost values.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from src.refgame.data.dataset import load_jsonl
from src.refgame.listeners.direct_rank import ImageOnlyIndexListener
from src.refgame.listeners.cost_aware import CostAwareListener
from src.refgame.eval.reporter import save_results, summarize
from src.refgame.utils.llm_client import openrouter
from src.refgame.data.schema import EvalRecord, Utterance
from src.refgame.metrics.core import brier_score, referential_entropy

DATASETS = [
    "data/scenes_6_none.jsonl",
    "data/scenes_6_force.jsonl",
    "data/scenes_8_none.jsonl",
    "data/scenes_8_force.jsonl",
    "data/scenes_10_none.jsonl",
    "data/scenes_10_force.jsonl",
]
HONEST_DIR   = Path("results/vllm_honest")
COST_VALUES  = [0.1, 0.25, 0.5]
N_WORKERS    = 6
MODEL        = "google/gemini-2.0-flash-001"


def load_utterance_cache(records_path: Path) -> dict[tuple[str, str], str]:
    """Return {(scene_id, speaker_type): utterance_text} from an existing records file."""
    cache: dict[tuple[str, str], str] = {}
    with open(records_path) as f:
        for line in f:
            r = json.loads(line)
            key = (str(r["scene_id"]), r["speaker_type"])
            cache[key] = r["utterance"]
    return cache


def run_index_listener(scenes, utt_cache, listener, cost_values):
    """
    For each (scene, speaker) pair in utt_cache, call the index listener once,
    then expand across cost_values (no extra LLM calls needed).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Collect unique (scene, speaker_type) keys that exist in the cache
    scene_by_id = {str(s.id): s for s in scenes}
    tasks = []
    for (scene_id, speaker_type), utt_text in utt_cache.items():
        if scene_id in scene_by_id:
            tasks.append((scene_id, speaker_type, utt_text))

    # Call listener in parallel (cached per (scene_id, speaker_type))
    posterior_cache: dict[tuple[str, str], tuple] = {}

    def _call(scene_id, speaker_type, utt_text):
        scene = scene_by_id[scene_id]
        utterance = Utterance(text=utt_text, speaker_type=speaker_type)
        out = listener.listen(scene, utterance)
        return (scene_id, speaker_type), out

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(_call, sid, stype, utxt): (sid, stype)
                for sid, stype, utxt in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                key, out = fut.result()
                posterior_cache[key] = out
                if i % 50 == 0:
                    logging.info(f"  io-index: {i}/{len(tasks)} calls done")
            except Exception as e:
                sid, stype = futs[fut]
                logging.warning(f"  Failed ({sid}, {stype}): {e}")

    # Expand across costs and build EvalRecords
    records = []
    for (scene_id, speaker_type), out in posterior_cache.items():
        scene = scene_by_id[scene_id]
        utt_text = utt_cache[(scene_id, speaker_type)]
        utterance = Utterance(text=utt_text, speaker_type=speaker_type)
        target_idx = None
        # find target_idx by scanning scene objects — stored in scene.target_idx if present
        # otherwise we need the original records; use a fallback: scenes store nothing,
        # so we need to read it from utt_cache source. We'll pass it through.
        for cost_c in cost_values:
            eu_commit = max(out.posterior)
            eu_ask    = 1.0 - cost_c
            action    = "ask" if eu_commit < eu_ask else "commit"
            correct   = None  # filled below once we have target_idx
            records.append(dict(
                scene_id=scene_id,
                speaker_type=speaker_type,
                listener_type=out.listener_type,
                cost_c=cost_c,
                utterance=utt_text,
                action=action,
                predicted_idx=out.predicted_idx,
                _out=out,
                _scene=scene,
            ))
    return records, posterior_cache


def main():
    client   = openrouter(model=MODEL)
    listener = ImageOnlyIndexListener(client=client)

    for ds_path in DATASETS:
        stem = Path(ds_path).stem
        records_path = HONEST_DIR / f"{stem}_records.jsonl"
        if not records_path.exists():
            logging.warning(f"No existing records at {records_path}, skipping")
            continue

        logging.info(f"=== {stem} ===")
        scenes = load_jsonl(ds_path)
        scene_by_id = {str(s.id): s for s in scenes}

        # Load existing records to get target_idx per (scene_id, speaker_type)
        # and utterance cache
        utt_cache: dict[tuple[str, str], str] = {}
        target_idx_map: dict[str, int] = {}  # scene_id -> target_idx (same across speakers)
        with open(records_path) as f:
            for line in f:
                r = json.loads(line)
                key = (str(r["scene_id"]), r["speaker_type"])
                utt_cache[key] = r["utterance"]
                target_idx_map[str(r["scene_id"])] = r["target_idx"]

        logging.info(f"  Loaded {len(utt_cache)} (scene, speaker) pairs from cache")

        # Unique (scene_id, speaker_type) — call listener once per pair
        unique_keys = list(utt_cache.keys())
        scene_ids_present = {k[0] for k in unique_keys if k[0] in scene_by_id}
        logging.info(f"  Scenes in dataset: {len(scene_ids_present)}")

        # Run index listener
        from concurrent.futures import ThreadPoolExecutor, as_completed

        posterior_cache: dict[tuple[str, str], object] = {}

        def _call(scene_id, speaker_type, utt_text):
            scene = scene_by_id[scene_id]
            utterance = Utterance(text=utt_text, speaker_type=speaker_type)
            out = listener.listen(scene, utterance)
            return (scene_id, speaker_type), out

        tasks = [(sid, stype, utt_cache[(sid, stype)])
                 for sid, stype in unique_keys if sid in scene_by_id]

        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            futs = {ex.submit(_call, sid, stype, utxt): (sid, stype)
                    for sid, stype, utxt in tasks}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    key, out = fut.result()
                    posterior_cache[key] = out
                    if i % 100 == 0 or i == len(tasks):
                        logging.info(f"  {i}/{len(tasks)} listener calls done")
                except Exception as e:
                    sid, stype = futs[fut]
                    logging.warning(f"  Failed ({sid}, {stype}): {e}")

        # Build EvalRecord objects
        new_records = []
        for (scene_id, speaker_type), out in posterior_cache.items():
            scene  = scene_by_id[scene_id]
            target_idx = target_idx_map.get(scene_id)
            if target_idx is None:
                continue
            utt_text = utt_cache[(scene_id, speaker_type)]
            utterance = Utterance(text=utt_text, speaker_type=speaker_type)
            bs = brier_score(out.posterior, target_idx)
            ent = referential_entropy(out.posterior)

            for cost_c in COST_VALUES:
                eu_commit = max(out.posterior)
                eu_ask    = 1.0 - cost_c
                action    = "ask" if eu_commit < eu_ask else "commit"
                correct   = (out.predicted_idx == target_idx) if action == "commit" else False

                rec = EvalRecord(
                    scene_id=scene_id,
                    speaker_type=speaker_type,
                    listener_type=out.listener_type,
                    cost_c=cost_c,
                    utterance=utt_text,
                    action=action,
                    predicted_idx=out.predicted_idx,
                    target_idx=target_idx,
                    correct=correct,
                    eu_commit=eu_commit,
                    eu_ask=eu_ask,
                    entropy=ent,
                    brier_score=bs,
                    min_desc_length=getattr(scene, "min_desc_length", None),
                    ambiguity_tier=getattr(scene, "ambiguity_tier", None),
                )
                new_records.append(rec)

        logging.info(f"  Generated {len(new_records)} new records for {stem}")

        # Merge into existing records file
        existing = []
        with open(records_path) as f:
            for line in f:
                existing.append(json.loads(line))

        # Append new records
        from dataclasses import asdict
        with open(records_path, "a") as f:
            for rec in new_records:
                f.write(json.dumps(asdict(rec)) + "\n")

        logging.info(f"  Merged into {records_path} ({len(existing)} existing + {len(new_records)} new)")

        # Rewrite summary
        all_raw = existing + [json.loads(json.dumps(asdict(r))) for r in new_records]

        # Print quick summary for new listener
        by_cost: dict[float, list] = defaultdict(list)
        for rec in new_records:
            d = asdict(rec)
            by_cost[d["cost_c"]].append(d)

        for cost_c in sorted(by_cost):
            rows = by_cost[cost_c]
            by_spk: dict[str, list] = defaultdict(list)
            for r in rows:
                by_spk[r["speaker_type"]].append(r)
            print(f"\n{stem}  c={cost_c:.2f}")
            for spk, recs in sorted(by_spk.items()):
                n = len(recs)
                acc = sum(r["correct"] for r in recs) / n
                ask = sum(r["action"] == "ask" for r in recs) / n
                cpa = acc - cost_c * ask
                print(f"  {spk:<55} acc={acc:.3f}  ask={ask:.1%}  cpa={cpa:.3f}")

    logging.info("Done.")


if __name__ == "__main__":
    main()
