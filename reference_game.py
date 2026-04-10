"""
Reference Game & Pragmatic Inference Experiments
=================================================
Two approaches for generating referring expressions:
  1. Classic RSA (Rational Speech Acts) over structured features
  2. LLM prompting (via OpenAI-compatible API or Anthropic API)

Each object is represented as:
  { "id": str, "x": float, "y": float, "shape": str, "color": str, "size": str }
"""

import itertools
import math
import json
from typing import Any

# ──────────────────────────────────────────────────────────────
# 1. Scene & Utterance Helpers
# ──────────────────────────────────────────────────────────────

SHAPES = ["circle", "square", "triangle", "diamond", "star"]
COLORS = ["red", "blue", "green", "yellow", "purple", "orange"]
SIZES  = ["small", "medium", "large"]
LOCATIONS = ["left", "right", "top", "bottom", "center",
             "top-left", "top-right", "bottom-left", "bottom-right"]


def loc_label(x: float, y: float) -> str:
    """Map (x, y) in [0,1]x[0,1] to a coarse spatial label."""
    col = "left" if x < 0.33 else ("right" if x > 0.66 else "center")
    row = "top"  if y < 0.33 else ("bottom" if y > 0.66 else "center")
    if row == col == "center":
        return "center"
    if row == "center":
        return col
    if col == "center":
        return row
    return f"{row}-{col}"


def generate_utterances(obj: dict, max_combo: int = 3) -> list[str]:
    """
    Enumerate candidate utterances from the object's features.
    Produces single-property ("the red one"), two-property ("the large red one"),
    and full descriptions ("the large red circle on the left").
    """
    props = {
        "color": obj["color"],
        "size":  obj["size"],
        "shape": obj["shape"],
        "location": loc_label(obj["x"], obj["y"]),
    }
    utterances = set()
    keys = list(props.keys())

    for r in range(1, max_combo + 1):
        for combo in itertools.combinations(keys, r):
            parts = [props[k] for k in combo if k != "shape"]
            shape_word = props["shape"] if "shape" in combo else "one"
            loc_part = ""
            if "location" in combo:
                loc_part = f" on the {props['location']}"
                parts = [p for p in parts if p != props["location"]]
            desc = " ".join(parts) + f" {shape_word}" + loc_part
            utterances.add(f"the {desc.strip()}")

    # Also add full description
    full = f"the {props['size']} {props['color']} {props['shape']} on the {props['location']}"
    utterances.add(full)
    return sorted(utterances)


# ──────────────────────────────────────────────────────────────
# 2. Classic RSA Model
# ──────────────────────────────────────────────────────────────

def literal_listener(utterance: str, objects: list[dict]) -> dict[str, float]:
    """
    L0: uniform over objects whose features are consistent with the utterance.
    """
    compatible = []
    for obj in objects:
        loc = loc_label(obj["x"], obj["y"])
        tokens = {obj["color"], obj["size"], obj["shape"], loc}
        # Check every content word in the utterance appears in object's tokens
        utt_words = set(utterance.replace("the ", "").replace(" on ", " ").split())
        # Remove filler words
        utt_words -= {"the", "one", "on"}
        if utt_words <= tokens:
            compatible.append(obj["id"])
    n = len(compatible) if compatible else 1
    return {obj["id"]: (1.0 / n if obj["id"] in compatible else 0.0)
            for obj in objects}


def pragmatic_speaker(target_id: str,
                      objects: list[dict],
                      utterances: list[str],
                      alpha: float = 4.0,
                      cost_weight: float = 0.1) -> list[tuple[str, float]]:
    """
    S1 pragmatic speaker: picks utterances that maximize informativeness
    (listener correctly identifying target) minus cost (utterance length).

    alpha:       rationality parameter (higher = more peaked)
    cost_weight: penalty per word (encourages brevity)
    """
    scores = {}
    for utt in utterances:
        l0 = literal_listener(utt, objects)
        informativeness = math.log(l0.get(target_id, 1e-10) + 1e-10)
        cost = cost_weight * len(utt.split())
        scores[utt] = alpha * informativeness - cost

    # Softmax
    max_s = max(scores.values())
    exp_scores = {u: math.exp(s - max_s) for u, s in scores.items()}
    total = sum(exp_scores.values())
    probs = {u: e / total for u, e in exp_scores.items()}
    return sorted(probs.items(), key=lambda x: -x[1])


def pragmatic_listener(utterance: str,
                       objects: list[dict],
                       alpha: float = 4.0,
                       cost_weight: float = 0.1) -> dict[str, float]:
    """
    L1 pragmatic listener: infers which object the speaker meant,
    reasoning about why the speaker chose this utterance.
    """
    # For each candidate object, compute S1 probability of this utterance
    obj_scores = {}
    for obj in objects:
        utts = generate_utterances(obj)
        s1 = dict(pragmatic_speaker(obj["id"], objects, utts, alpha, cost_weight))
        obj_scores[obj["id"]] = s1.get(utterance, 1e-10)

    total = sum(obj_scores.values())
    return {oid: s / total for oid, s in obj_scores.items()}


def rsa_refer(target_id: str,
              objects: list[dict],
              alpha: float = 4.0,
              cost_weight: float = 0.1,
              top_k: int = 5) -> list[tuple[str, float]]:
    """
    Full RSA pipeline: generate utterances for target, rank by S1.
    Returns top_k (utterance, probability) pairs.
    """
    target = next(o for o in objects if o["id"] == target_id)
    utterances = generate_utterances(target)
    ranked = pragmatic_speaker(target_id, objects, utterances, alpha, cost_weight)
    return ranked[:top_k]


# ──────────────────────────────────────────────────────────────
# 3. LLM Prompting Approach
# ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT_SPEAKER = """\
You are playing a reference game. You are the SPEAKER.

You see a scene with several objects. Each object has attributes:
  shape, color, size, and (x, y) position.

Your job: produce a short natural-language referring expression that
uniquely picks out the TARGET object so a listener can identify it
among the distractors. Be concise but unambiguous.

Rules:
- Use only visual properties visible in the scene.
- Prefer brevity: omit attributes that don't help distinguish the target.
- Include spatial terms (left, right, top, bottom) only if needed.
- Output ONLY the referring expression, nothing else.
"""

SYSTEM_PROMPT_LISTENER = """\
You are playing a reference game. You are the LISTENER.

You see a scene with several objects. Each object has attributes:
  shape, color, size, and (x, y) position.

The speaker has produced a referring expression. Your job: determine
which object (by ID) the speaker is most likely referring to.

Output a JSON object: {"target_id": "<id>", "confidence": <0-1>, "reasoning": "<brief>"}
"""

SYSTEM_PROMPT_PRAGMATIC = """\
You are a pragmatic speaker in a reference game, modeled after
Rational Speech Acts (RSA) theory.

You see a scene with objects. Your goal: produce a referring expression
for the TARGET that a pragmatic listener would correctly resolve.

Think step by step:
1. List features of the target.
2. List features of each distractor.
3. Identify which features UNIQUELY distinguish the target.
4. Construct the shortest expression using only distinguishing features.
5. If multiple features are needed, combine them naturally.

Output format:
REASONING: <your step-by-step analysis>
EXPRESSION: <your final referring expression>
"""


def format_scene(objects: list[dict], target_id: str | None = None) -> str:
    """Format a scene for an LLM prompt."""
    lines = ["SCENE:"]
    for obj in objects:
        loc = loc_label(obj["x"], obj["y"])
        marker = " ← TARGET" if obj["id"] == target_id else ""
        lines.append(
            f'  [{obj["id"]}] {obj["size"]} {obj["color"]} {obj["shape"]} '
            f'at ({obj["x"]:.2f}, {obj["y"]:.2f}) [{loc}]{marker}'
        )
    return "\n".join(lines)


def build_speaker_prompt(objects: list[dict], target_id: str,
                         pragmatic: bool = False) -> list[dict]:
    """Build chat messages for speaker prompting."""
    system = SYSTEM_PROMPT_PRAGMATIC if pragmatic else SYSTEM_PROMPT_SPEAKER
    scene = format_scene(objects, target_id)
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": scene},
    ]


def build_listener_prompt(objects: list[dict], utterance: str) -> list[dict]:
    """Build chat messages for listener prompting."""
    scene = format_scene(objects)
    return [
        {"role": "system", "content": SYSTEM_PROMPT_LISTENER},
        {"role": "user",   "content": f"{scene}\n\nSPEAKER SAYS: \"{utterance}\""},
    ]


def build_evaluation_prompt(objects: list[dict], target_id: str,
                             utterance: str) -> list[dict]:
    """
    Prompt an LLM to evaluate whether a referring expression is
    sufficient, over-specified, or ambiguous.
    """
    scene = format_scene(objects, target_id)
    return [
        {"role": "system", "content": (
            "You are an evaluator for a reference game. Given a scene, a target "
            "object, and a referring expression, assess:\n"
            "1. SUFFICIENT: Does it uniquely identify the target?\n"
            "2. MINIMAL: Does it use only necessary attributes?\n"
            "3. NATURAL: Does it sound like something a human would say?\n"
            "Output JSON: {\"sufficient\": bool, \"minimal\": bool, "
            "\"natural\": bool, \"score\": 0-10, \"feedback\": \"...\"}"
        )},
        {"role": "user", "content": (
            f"{scene}\n\nEXPRESSION: \"{utterance}\"\n"
            f"TARGET: {target_id}"
        )},
    ]


# ──────────────────────────────────────────────────────────────
# 4. API Callers (Anthropic + OpenAI-compatible)
# ──────────────────────────────────────────────────────────────

def call_anthropic(messages: list[dict], model: str = "claude-sonnet-4-20250514",
                   api_key: str | None = None, max_tokens: int = 512) -> str:
    """Call Anthropic Messages API. Requires `anthropic` package."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)  # uses ANTHROPIC_API_KEY env var if None
    system = ""
    chat = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        else:
            chat.append(m)
    resp = client.messages.create(
        model=model, max_tokens=max_tokens, system=system, messages=chat
    )
    return resp.content[0].text


def call_openai(messages: list[dict], model: str = "gpt-4o",
                api_key: str | None = None, max_tokens: int = 512) -> str:
    """Call OpenAI-compatible chat completions API."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens, temperature=0.3
    )
    return resp.choices[0].message.content


# ──────────────────────────────────────────────────────────────
# 5. Experiment Runner
# ──────────────────────────────────────────────────────────────

def run_experiment(scene: list[dict],
                   target_id: str,
                   alpha: float = 4.0,
                   cost_weight: float = 0.1,
                   llm_caller: Any = None,
                   verbose: bool = True) -> dict:
    """
    Run a full experiment: RSA + optional LLM speaker/listener/evaluator.

    Parameters
    ----------
    scene       : list of object dicts
    target_id   : id of the target object
    alpha       : RSA rationality
    cost_weight : RSA brevity cost
    llm_caller  : function(messages) -> str, e.g. call_anthropic or call_openai
    verbose     : print results

    Returns
    -------
    dict with RSA results, and LLM results if llm_caller provided.
    """
    results = {"scene": scene, "target_id": target_id}

    # --- RSA ---
    rsa_ranked = rsa_refer(target_id, scene, alpha, cost_weight)
    results["rsa_top"] = rsa_ranked
    if verbose:
        print("=" * 60)
        print("SCENE:")
        for obj in scene:
            marker = " ← TARGET" if obj["id"] == target_id else ""
            print(f"  [{obj['id']}] {obj['size']} {obj['color']} {obj['shape']} "
                  f"at ({obj['x']:.2f}, {obj['y']:.2f}){marker}")
        print(f"\nRSA S1 top expressions (α={alpha}, cost={cost_weight}):")
        for utt, prob in rsa_ranked:
            print(f"  {prob:.4f}  {utt}")

    # --- LLM ---
    if llm_caller:
        # Naive speaker
        msgs = build_speaker_prompt(scene, target_id, pragmatic=False)
        naive_expr = llm_caller(msgs)
        results["llm_naive"] = naive_expr

        # Pragmatic (chain-of-thought) speaker
        msgs = build_speaker_prompt(scene, target_id, pragmatic=True)
        prag_expr = llm_caller(msgs)
        results["llm_pragmatic"] = prag_expr

        # Listener verification on both
        for label, expr in [("naive", naive_expr), ("pragmatic", prag_expr)]:
            # Extract just the expression if pragmatic format
            clean = expr.split("EXPRESSION:")[-1].strip().strip('"') if "EXPRESSION:" in expr else expr.strip().strip('"')
            msgs = build_listener_prompt(scene, clean)
            listener_resp = llm_caller(msgs)
            results[f"llm_listener_{label}"] = listener_resp

            # Evaluation
            msgs = build_evaluation_prompt(scene, target_id, clean)
            eval_resp = llm_caller(msgs)
            results[f"llm_eval_{label}"] = eval_resp

        if verbose:
            print(f"\nLLM naive speaker:     {results['llm_naive']}")
            print(f"LLM pragmatic speaker: {results['llm_pragmatic']}")
            print(f"\nListener (naive):      {results['llm_listener_naive']}")
            print(f"Listener (pragmatic):  {results['llm_listener_pragmatic']}")
            print(f"\nEval (naive):          {results['llm_eval_naive']}")
            print(f"Eval (pragmatic):      {results['llm_eval_pragmatic']}")

    if verbose:
        print("=" * 60)
    return results


# ──────────────────────────────────────────────────────────────
# 6. Example Scenes
# ──────────────────────────────────────────────────────────────

EXAMPLE_SCENES = [
    {
        "name": "Color-only distinguishes",
        "objects": [
            {"id": "A", "x": 0.2, "y": 0.5, "shape": "circle", "color": "red",   "size": "large"},
            {"id": "B", "x": 0.5, "y": 0.5, "shape": "circle", "color": "blue",  "size": "large"},
            {"id": "C", "x": 0.8, "y": 0.5, "shape": "circle", "color": "green", "size": "large"},
        ],
        "target": "A",
    },
    {
        "name": "Needs two features",
        "objects": [
            {"id": "A", "x": 0.2, "y": 0.2, "shape": "square",   "color": "red",  "size": "small"},
            {"id": "B", "x": 0.8, "y": 0.2, "shape": "square",   "color": "red",  "size": "large"},
            {"id": "C", "x": 0.5, "y": 0.8, "shape": "triangle", "color": "blue", "size": "small"},
        ],
        "target": "A",
    },
    {
        "name": "Spatial reference needed",
        "objects": [
            {"id": "A", "x": 0.1, "y": 0.1, "shape": "circle", "color": "red", "size": "large"},
            {"id": "B", "x": 0.9, "y": 0.9, "shape": "circle", "color": "red", "size": "large"},
        ],
        "target": "A",
    },
    {
        "name": "Complex scene",
        "objects": [
            {"id": "A", "x": 0.1, "y": 0.1, "shape": "circle",   "color": "red",    "size": "small"},
            {"id": "B", "x": 0.5, "y": 0.1, "shape": "circle",   "color": "red",    "size": "large"},
            {"id": "C", "x": 0.9, "y": 0.1, "shape": "square",   "color": "red",    "size": "small"},
            {"id": "D", "x": 0.1, "y": 0.9, "shape": "circle",   "color": "blue",   "size": "small"},
            {"id": "E", "x": 0.9, "y": 0.9, "shape": "triangle", "color": "green",  "size": "large"},
        ],
        "target": "A",
    },
]


# ──────────────────────────────────────────────────────────────
# 7. Main
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reference game experiments")
    parser.add_argument("--llm", choices=["anthropic", "openai", "none"], default="none",
                        help="Which LLM API to use (default: none, RSA only)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name override")
    parser.add_argument("--alpha", type=float, default=4.0,
                        help="RSA rationality parameter")
    parser.add_argument("--cost", type=float, default=0.1,
                        help="RSA cost weight per word")
    parser.add_argument("--scene", type=int, default=None,
                        help="Run only scene N (0-indexed)")
    args = parser.parse_args()

    # Set up LLM caller
    llm = None
    if args.llm == "anthropic":
        model = args.model or "claude-sonnet-4-20250514"
        llm = lambda msgs, m=model: call_anthropic(msgs, model=m)
    elif args.llm == "openai":
        model = args.model or "gpt-4o"
        llm = lambda msgs, m=model: call_openai(msgs, model=m)

    scenes = EXAMPLE_SCENES
    if args.scene is not None:
        scenes = [scenes[args.scene]]

    all_results = []
    for sc in scenes:
        print(f"\n{'─' * 60}")
        print(f"EXPERIMENT: {sc['name']}")
        print(f"{'─' * 60}")
        res = run_experiment(
            scene=sc["objects"],
            target_id=sc["target"],
            alpha=args.alpha,
            cost_weight=args.cost,
            llm_caller=llm,
        )
        all_results.append(res)

    # Save results
    with open("results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to results.json")
