# DueDiligenceAI Corpus — Document Index and Evaluation Question Set

> Do not ingest this file into `data/raw/` (it contains no source facts and would pollute retrieval). Move it to `data/evaluation/`.

## Document Index

| Filename | Company | Type | Effective Date | Key Facts |
|----------|---------|------|---------------|-----------|
| nexacore_annual_report_fy2023.md | NexaCore | Annual Report FY2023 | 2024-03-15 | Revenue $14.2B (+38%); GM 68.3%; NI $4.1B; EPS $1.64; China 18%; X1 80 GB HBM3; SilicaFab through Dec 2027; 12,400 employees |
| nexacore_annual_report_fy2024.md | NexaCore | Annual Report FY2024 | 2025-03-12 | Revenue $29.5B (+108%); GM 71.9%; NI $9.8B; X2 launch (160 GB HBM3e, 2.4×); DataSync $2.3B; China 20% ($5.9B); top-3 = 52%; guidance $58–64B |
| nexacore_annual_report_fy2025.md | NexaCore | Annual Report FY2025 | 2026-03-10 | Revenue $61.2B (+107%); GM 76.0%; NI $22.1B; EPS $8.84; China 8% ($4.9B); top-3 = 58% ($27.9B); $12B buyback; Project Diversify; DOJ inquiry; guidance $85–95B |
| nexacore_q4_fy2025_earnings_call.md | NexaCore | Earnings Call Q4 FY2025 | 2026-02-26 | Q4 revenue $17.8B (+115%); Q4 GM 77.2%; Q4 EPS $2.52; **China 9% (conflicts with 8%)**; China FY2026 5–6%; unsupported "end of this decade" claim |
| quantumedge_annual_report_fy2024.md | QuantumEdge | Annual Report FY2024 | 2025-04-05 | Revenue $18.7B (+62%); GM 61.1%; NI $2.4B; 1,800 customers; GlobalChip; guidance $26–30B; litigation addendum (Nov 2025) |
| quantumedge_annual_report_fy2025.md | QuantumEdge | Annual Report FY2025 | 2026-04-08 | Revenue $28.3B (+51%); GM 65.0%; NI $4.8B; QuantumCloud $1.4B / 180,000 customers; 7 patents; Case 1:2025cv08841; guidance $38–42B |
| ai_semiconductor_industry_report_2025.md | Industry (Meridian) | Industry Report | 2026-01-15 | Market $142B; NexaCore 43%; **planted claim: adjusted GM ~71%**; Taiwan risk; HBM4 2.4B vs 4.2B units |

## Evaluation Questions (50)

### Factual (15)

1. Who is the CEO of NexaCore Technologies, and when was she appointed?
2. Where is SilicaFab based, and until when does the SilicaFab agreement run?
3. How much did NexaCore pay to acquire DataSync Inc, and how many employees did DataSync have?
4. What memory and relative performance does the NexaCore X2 Accelerator have?
5. What are the specifications of the NexaCore X1 Accelerator?
6. What is NexaCore X3's memory type and capacity, and what is its performance gain over X2?
7. Which two Korean companies supply HBM memory to NexaCore?
8. What is NexaOS DataBridge and how many enterprise customers had adopted it by FY2025?
9. What is the case number of QuantumEdge's patent suit against NexaCore, and how many patents are claimed?
10. What is Project Diversify?
11. Who manufactures QuantumEdge's chips, and where?
12. What is QuantumCloud, and how many customers did it have at the end of FY2025?
13. What government inquiry did NexaCore receive in October 2025?
14. What was NexaCore's Q4 FY2025 diluted EPS?
15. Which research firm published the 2025 AI semiconductor market review, and what rating did it give QuantumEdge?

### Numerical / Calculation (10)

16. Verify NexaCore's FY2025 gross margin from gross profit and revenue.
17. What dollar amount is 58% of NexaCore's FY2025 Data Center revenue?
18. By what percentage did NexaCore's R&D expense grow from FY2023 to FY2025?
19. What was NexaCore's net cash at the end of FY2025?
20. What share of NexaCore's FY2025 revenue came from Data Center?
21. What was NexaCore's R&D expense as a percentage of revenue in each of the three years?
22. By how much (in $ and %) did NexaCore's China revenue change from FY2024 to FY2025?
23. What is QuantumEdge's FY2025 operating margin, computed from operating income and revenue?
24. How much did NexaCore's capacity reservation payments to SilicaFab increase between FY2024 and FY2025?
25. What was NexaCore's Q4 FY2025 revenue as a share of full-year revenue?

### Comparison — NexaCore vs QuantumEdge (10)

26. Compare FY2025 revenue and revenue growth for the two companies.
27. Compare FY2025 gross margins and explain the stated reasons for the gap.
28. Compare operating margins in FY2024 and FY2025.
29. Compare R&D as a percentage of revenue for FY2025.
30. Compare manufacturing locations and geopolitical exposure.
31. Compare free cash flow and EPS in FY2025.
32. Compare each company's FY2026 guidance.
33. Compare their software businesses (NexaOS DataBridge vs EdgeNet/QuantumCloud).
34. Compare their HBM4 positions.
35. Compare employee counts and engineering headcount in FY2025.

### Temporal (8)

36. How did NexaCore's gross margin change from FY2023 to FY2025, and what drove it?
37. How did China's share of NexaCore revenue evolve across FY2023–FY2025, and why?
38. How did NexaCore's top-3 customer concentration change between FY2024 and FY2025?
39. How did NexaCore's employee count change from FY2023 to FY2025?
40. How did NexaCore's segment reporting change between FY2023 and FY2024?
41. How did QuantumEdge's revenue growth rate change from FY2024 to FY2025?
42. How did NexaCore's stated primary risk change between FY2023, FY2024, and FY2025?
43. How did NexaCore's cash and debt positions change over the three years?

### Multi-hop (5)

44. If a 60-day SilicaFab disruption would remove roughly $10B of quarterly revenue, what share is that of Q4 FY2025 revenue, and what mitigation is underway?
45. Which company's HBM4 allocation is exposed to competition from the other, and what are each company's disclosed allocation positions?
46. How does Meridian's market-share figure for NexaCore (43%) relate to the 72% figure in company filings?
47. What is the relationship between the patent litigation, the DOJ inquiry, and NexaCore's market position?
48. Combine the export-control timeline and the China revenue figures to explain NexaCore's FY2026 China outlook.

### Adversarial / Validation (2)

49. What percentage of NexaCore's FY2025 revenue came from China? (Annual report says 8%; earnings call says 9%: should flag conflict.)
50. What is NexaCore's true FY2025 gross margin — 76% or 71%? (Meridian's 71% claim conflicts with the audited 76% and should be attributed and flagged. Also test: "Will NexaCore maintain market leadership through the end of the decade?" should not be answered as fact.)
