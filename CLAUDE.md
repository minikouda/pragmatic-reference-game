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

Code (to be added) will likely include:
- Dataset generator (synthetic JSON "Mini-World" scenes with controllable referential entropy)
which will create scenes with varying numbers of objects and attributes to manipulate ambiguity levels.
Can generate datasets with different parameters. Like if target is overlapping with other objects, or if minimum number of features to uniquely identify the target is high, etc.

- Speaker
Three speaker: LLM Speaker (using GPT-4 to generate utterances), Literal Speaker (using a simple rule-based approach to generate literal descriptions), and RSA Speaker (using RSA to generate pragmatic utterances that consider the listener's perspective).

- Listener
Two listeners: Literal Listener (using a simple rule-based approach to interpret utterances) and RSA Listener (using RSA to interpret utterances by considering the speaker's perspective and the context). And a Cost-Aware Listener that decides whether to ask a clarifying question based on the expected utility of asking vs. guessing.

- Evaluation scripts