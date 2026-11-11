# Phase 5 Test Commands

> Run all commands from the project root directory.

```bash
## Unit Tests
pytest tests/planning/ -v
pytest tests/planning/test_analyzer.py -v
pytest tests/planning/test_planner.py -v
pytest tests/planning/test_validator.py -v
pytest tests/orchestration/ -v
pytest tests/orchestration/test_executor.py -v
pytest tests/orchestration/test_graph.py -v

## Inspect query plans
python scripts/inspect_plan.py "What is NexaCore's FY2025 gross margin?"
python scripts/inspect_plan.py "Compare NexaCore and QuantumEdge revenue in FY2025"
python scripts/inspect_plan.py "How did NexaCore's risk profile change from FY2023 to FY2025?"
python scripts/inspect_plan.py "NexaCore's main risks and revenue trends"
python scripts/inspect_plan.py "Who manufactures NexaCore's X2 chips and where are they located?"

## Inspect full research traces
python scripts/inspect_research.py "Compare NexaCore and QuantumEdge gross margins"
python scripts/inspect_research.py "How did NexaCore's China exposure change from FY2024 to FY2025?"

## Evaluate planning quality (50 questions)
python scripts/evaluate_planning.py

## Expected metrics:
# Classification Accuracy: ~82%
# Task Count Accuracy: ~84%
# Plan Validity Rate: 100%
# Over-decomposition Rate: ~6%

## Streamlit (use Research Agent mode in sidebar)
streamlit run ui.py
```
