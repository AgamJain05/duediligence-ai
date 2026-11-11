# Phase 6 Test Commands

Run these manually. Do NOT auto-execute.

## Setup (run once)

> Run all commands from the project root directory.

```bash
# Apply Phase 6 migration
alembic upgrade head

# Dry-run metric ingestion (no DB writes)
python scripts/ingest_metrics.py --dry-run

# Ingest metrics into DB
python scripts/ingest_metrics.py
```

## Inspect Metrics

```bash
# List all companies
python scripts/inspect_metrics.py --list-companies

# List canonical metric names
python scripts/inspect_metrics.py --list-metrics

# NexaCore revenue all years
python scripts/inspect_metrics.py --company NexaCore --metric revenue --all-years

# NexaCore R&D all years
python scripts/inspect_metrics.py --company NexaCore --metric rd_expense --all-years

# NexaCore gross margin all years
python scripts/inspect_metrics.py --company NexaCore --metric gross_margin --all-years

# CONFLICT: NexaCore China share FY2025 — should show 2 rows (8% and 9%)
python scripts/inspect_metrics.py --company NexaCore --metric regional_revenue_share --year 2025 --dim-type REGION --dim-value China

# QuantumEdge FY2024 and FY2025 gross margin
python scripts/inspect_metrics.py --company QuantumEdge --metric gross_margin --all-years

# NexaCore Data Center segment FY2025
python scripts/inspect_metrics.py --company NexaCore --metric segment_revenue --year 2025 --dim-type SEGMENT
```

## Inspect Tools

```bash
# List all registered tools
python scripts/inspect_tools.py

# Calculator: percent change NexaCore revenue FY2024->FY2025
python scripts/inspect_tools.py --tool calculate_metric --args '{"operation":"percent_change","inputs":[{"label":"FY2024","value":"29.5","unit":"USD"},{"label":"FY2025","value":"61.2","unit":"USD"}]}'

# Calculator: basis point change QuantumEdge gross margin
python scripts/inspect_tools.py --tool calculate_metric --args '{"operation":"basis_point_change","inputs":[{"label":"FY2024","value":"61.1","unit":"PERCENT"},{"label":"FY2025","value":"65.0","unit":"PERCENT"}]}'

# Calculator: Data Center share of total FY2025
python scripts/inspect_tools.py --tool calculate_metric --args '{"operation":"share_of_total","inputs":[{"label":"DataCenter","value":"48.1","unit":"USD"},{"label":"Total","value":"61.2","unit":"USD"}]}'

# Calculator: net cash
python scripts/inspect_tools.py --tool calculate_metric --args '{"operation":"difference","inputs":[{"label":"cash","value":"26.3","unit":"USD"},{"label":"debt","value":"3.5","unit":"USD"}]}'

# Financial data: NexaCore revenue FY2025
python scripts/inspect_tools.py --tool search_financial_data --args '{"company":"NexaCore Technologies","metric":"revenue","fiscal_year":2025}'

# Financial data: CONFLICT — China share FY2025
python scripts/inspect_tools.py --tool search_financial_data --args '{"company":"NexaCore Technologies","metric":"regional_revenue_share","fiscal_year":2025,"dimension_value":"China"}'

# Financial data: unknown metric (should return EMPTY with suggestions)
python scripts/inspect_tools.py --tool search_financial_data --args '{"company":"NexaCore Technologies","metric":"ebitda","fiscal_year":2025}'

# Financial data: unknown company (should return ERROR with known companies)
python scripts/inspect_tools.py --tool search_financial_data --args '{"company":"IntelliChip Corp","metric":"revenue","fiscal_year":2025}'

# Period compare: NexaCore gross margin FY2023 vs FY2025
python scripts/inspect_tools.py --tool compare_periods --args '{"company":"NexaCore Technologies","metric":"gross_margin","period_a":"2023","period_b":"2025"}'

# Period compare: QuantumEdge revenue FY2024 vs FY2025
python scripts/inspect_tools.py --tool compare_periods --args '{"company":"QuantumEdge Corp","metric":"revenue","period_a":"2024","period_b":"2025"}'
```

## Unit Tests (Phase 6)

```bash
# All Phase 6 tests
pytest tests/financial/ -v
pytest tests/tools/ -v

# Specific test modules
pytest tests/financial/test_parsing.py -v
pytest tests/financial/test_aliases.py -v
pytest tests/financial/test_repository.py -v
pytest tests/financial/test_ingest.py -v
pytest tests/tools/test_calculator.py -v
pytest tests/tools/test_financial_data.py -v
pytest tests/tools/test_period_compare.py -v
pytest tests/tools/test_registry.py -v
pytest tests/planning/test_models_p6.py -v
pytest tests/planning/test_analyzer_p6.py -v
pytest tests/orchestration/test_executor_p6.py -v
```

## Evaluation

```bash
# Run evaluation against tool_questions.json
python scripts/evaluate_tools.py

# Run specific category
python scripts/evaluate_tools.py --category METRICS_ONLY
python scripts/evaluate_tools.py --category METRICS_PLUS_CALC
python scripts/evaluate_tools.py --category HYBRID
python scripts/evaluate_tools.py --category conflict

# Compare Method A vs B vs C
python scripts/compare_phases.py --mode phase4
python scripts/compare_phases.py --mode phase5
python scripts/compare_phases.py --mode phase6
```

## Key Experiments to Verify

```bash
# Experiment 1: Conflict detection — both rows exist
python scripts/inspect_metrics.py --company NexaCore --metric regional_revenue_share --year 2025 --dim-value China
# Expected: 2 rows (8% annual report, 9% earnings call)

# Experiment 2: Idempotency — run twice, count stays the same
python scripts/ingest_metrics.py
python scripts/ingest_metrics.py
python scripts/inspect_metrics.py --company NexaCore --metric revenue --all-years
# Expected: same row count after second run

# Experiment 3: Calculator precision
python scripts/inspect_tools.py --tool calculate_metric --args '{"operation":"percent_change","inputs":[{"label":"old","value":"9.8","unit":"USD"},{"label":"new","value":"22.1","unit":"USD"}]}'
# Expected: (22.1 - 9.8) / 9.8 * 100 = 125.51%

# Experiment 4: Division by zero
python scripts/inspect_tools.py --tool calculate_metric --args '{"operation":"ratio","inputs":[{"label":"a","value":"100","unit":"USD"},{"label":"b","value":"0","unit":"USD"}]}'
# Expected: ERROR with code DIVISION_BY_ZERO

# Experiment 5: Unknown metric returns suggestions
python scripts/inspect_tools.py --tool search_financial_data --args '{"company":"NexaCore Technologies","metric":"earnings"}'
# Expected: EMPTY with suggestions including eps_diluted, net_income
```
