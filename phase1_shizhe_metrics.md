# Phase 1 — Brier Score & ECE Definitions for the Reference Game Task
**Author:** Shizhe Zhang

---

## Task Setup

In our reference game, each instance consists of:
- A scene with **N candidate objects** `{t_1, ..., t_N}`, each with discrete attributes (color, shape, size, position)
- An utterance `u` that refers to one target object `t*`
- A model that outputs a probability distribution `P(t_i | u)` over all N objects

The model must either **commit** (predict argmax) or **ask** a clarifying question, depending on the expected utility.

---

## Brier Score (Multi-class adaptation)

### Definition

For a single instance with N candidate objects, let:
- `p_i = P(t_i | u)` be the model's predicted probability for object i
- `y_i = 1` if `t_i` is the true target, else `y_i = 0` (one-hot ground truth)

The **multi-class Brier Score** for one instance is:

```
BS = (1/N) * sum_{i=1}^{N} (p_i - y_i)^2
```

Averaged over M instances:

```
BS_total = (1/M) * sum_{j=1}^{M} (1/N_j) * sum_{i=1}^{N_j} (p_i^(j) - y_i^(j))^2
```

### Interpretation

- **BS = 0**: perfect predictions (all probability mass on the correct object)
- **BS = 1**: maximally wrong (all probability mass on incorrect objects)
- A **uniform model** (p_i = 1/N for all i) scores BS = (N-1)/N^2 ≈ 1/N for large N

### Why this matters for our project

Brier score directly measures the quality of the probability distribution `P(t | u)`, which is the input to our expected utility calculation. If BS is high, our utility estimates are unreliable — this is an important finding about whether RSA-based inference actually helps calibration.

### Worked Example

Scene: 3 objects — red square, blue circle, red triangle. Utterance: "the red one."

| Object       | p_i (model) | y_i (truth) | (p_i - y_i)^2 |
|--------------|-------------|-------------|----------------|
| red square   | 0.50        | 1           | 0.25           |
| blue circle  | 0.10        | 0           | 0.01           |
| red triangle | 0.40        | 0           | 0.16           |

**BS = (1/3)(0.25 + 0.01 + 0.16) = 0.14**

A low Brier score here confirms the model is well-calibrated: it spread probability between the two red objects (ambiguity) while ignoring the blue circle.

---

## Expected Calibration Error (ECE)

### Definition

ECE measures whether a model's confidence matches its actual accuracy. We adapt ECE to the multi-object setting using the **top predicted probability** per instance.

**Procedure:**
1. For each instance j, extract: `conf_j = max_i P(t_i | u)` and `correct_j = 1` if argmax is the true target, else 0
2. Bin instances into B equal-width bins by confidence: `[0, 1/B), [1/B, 2/B), ..., [(B-1)/B, 1]` (default B=10)
3. For each non-empty bin b:
   - `acc(b) = mean(correct_j for j in b)`
   - `conf(b) = mean(conf_j for j in b)`
4. ECE is the weighted average gap:

```
ECE = sum_{b=1}^{B} (|b| / M) * |acc(b) - conf(b)|
```

where `|b|` is the number of instances in bin b and M is total instances.

### Interpretation

- **ECE = 0**: perfectly calibrated (90% confidence → correct 90% of the time)
- **ECE > 0.10**: poorly calibrated; utility calculations using these probabilities are unreliable
- A model that always outputs uniform probabilities will have ECE ≈ 0 (trivially calibrated but useless)

### Why this matters for our project

Our expected utility formula is:

```
E[U_commit] = P(correct | u, commit) - 0
E[U_ask]    = P(correct | u, ask) - c
```

where `P(correct | u, commit) = max_i P(t_i | u)`. If ECE is high, these expected utilities are wrong, and the system will ask/commit at the wrong times. Poor ECE is a diagnostic finding, not just a failure — it tells us RSA-based probability estimates need recalibration.

### Worked Example

10 instances, confidence vs. correctness:

| Bin [conf range] | Instances | avg conf | avg acc | gap  | weight |
|------------------|-----------|----------|---------|------|--------|
| [0.3, 0.5)       | 3         | 0.40     | 0.33    | 0.07 | 0.3    |
| [0.5, 0.7)       | 4         | 0.60     | 0.75    | 0.15 | 0.4    |
| [0.7, 1.0]       | 3         | 0.85     | 0.67    | 0.18 | 0.3    |

**ECE = 0.3(0.07) + 0.4(0.15) + 0.3(0.18) = 0.021 + 0.060 + 0.054 = 0.135**

This is borderline poor — the model is overconfident in the high-confidence bin (85% confident but only 67% accurate).

---

## Summary

| Metric | Computed over | Good value | Bad value |
|--------|--------------|------------|-----------|
| Brier Score | Full distribution P(t_i \| u) per instance | Near 0 | Near 1 |
| ECE | Top confidence per instance, binned | Near 0 | > 0.10 |

Both metrics are prerequisites for validating the expected utility framework: if `P(t | u)` is poorly calibrated (high ECE) or diffuse (high BS), the cost-aware clarification policy cannot make reliable decisions.
