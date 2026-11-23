# Vision-Language Model Spatial Grounding Evaluator

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2.10-green.svg)](https://docs.pydantic.dev/)
[![Tests Passing](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)](test_evaluator.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

A high-precision evaluation framework for benchmarking **Vision-Language Model (VLM)** spatial grounding predictions against ground-truth bounding box coordinates. Enforces strict Pydantic v2 schema validation, calculates exact Intersection-over-Union (IoU), and generates normalized spatial accuracy rubrics.

---

## Architecture Overview

```
                          ┌───────────────────────────┐
                          │   VLM Inference Output    │
                          │ [ymin, xmin, ymax, xmax]  │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │   Pydantic v2 Validator   │
                          │   Strict Coordinate Lock  │
                          └─────────────┬─────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
        ┌───────────────────────┐               ┌───────────────────────┐
        │  Spatial IoU Compute  │               │   Boundary & Format   │
        │  Intersection / Union │               │   Integrity Sentinel  │
        └───────────┬───────────┘               └───────────┬───────────┘
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        ▼
                          ┌───────────────────────────┐
                          │ Normalized Grounding Score│
                          │ Strict JSON Rubric Export │
                          └───────────────────────────┘
```

---

## Key Capabilities

- **Pydantic v2 Schema Enforcement**: Normalizes box inputs to `[0, 1]` or pixel bounds with strict invariant checks (`ymin < ymax`, `xmin < xmax`).
- **Deterministic IoU Computation**: Calculates exact spatial intersection over union across arbitrary box sets.
- **Categorical Precision Scoring**: Evaluates label classification match alongside spatial localization overlap.
- **Resilient CLI & Batch Ingestion**: Evaluates single inference outputs or high-throughput JSON datasets with structured progress summaries.

---

## Quickstart

### 1. Requirements

```bash
pip install pydantic pytest
```

### 2. Run Single Sample Evaluation

```bash
python main.py --demo
```

### 3. Evaluate Dataset Batch

```bash
python main.py --dataset sample_dataset.json --iou-threshold 0.50
```

---

## Running Test Suite

```bash
pytest test_evaluator.py -v
```

All 5 test suites validate box normalization, boundary invariants, zero-division edge cases, and schema conformance.

---

## Author

**Shivansh Dimri**  
Associate AI Engineer — Python Engineering · AI Evaluation · Data Systems · Automation  
Website: [https://shivanshdimri.in](https://shivanshdimri.in)  
GitHub: [@shiv677](https://github.com/shiv677)  
Email: [dimrishiv933@gmail.com](mailto:dimrishiv933@gmail.com)
