# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CS 288 Spring 2026 final project by Shizhe Zhang. The project investigates **cost-aware clarification in reference games** using Rational Speech Act (RSA) theory and expected utility calculations to determine when an LLM-based agent should ask a clarifying question vs. commit to an answer.

Core idea: model clarification as a cost-utility optimization at inference time (training-free), using referential entropy `H(T | u)` to trigger clarification when ambiguity is high and the cost of a wrong answer outweighs the cost of asking.

## Research Context

- **Task:** Reference game where an agent must identify a target object from a scene given an utterance
- **Key decision:** Ask a clarifying question (cheap but has a cost `c`) vs. commit to a guess (risky under high ambiguity)
- **Framework:** RSA-based pragmatic inference + expected utility: `Reward = Accuracy - c × (Questions Asked)`
- **Metrics:** Cost-Penalized Accuracy, Clarification Frequency (vs. entropy/cost), ECE, Brier Score

## Repository Structure

```
final_project/
  report/          # LaTeX report (note.tex / note.pdf)
```

Code (to be added) will likely include:
- Dataset generator (synthetic JSON "Mini-World" scenes with controllable referential entropy)
- Literal Listener baseline and heuristic string-match baseline
- RSA + expected utility clarification policy
- Evaluation scripts

## Report

Compile the report with:
```bash
cd report && pdflatex note.tex
```
