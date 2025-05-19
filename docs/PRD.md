# Product-Requirements Document – VitronMax  v0.1

## 1. Problem
Speed + explainability missing in BBB in-silico screening at scale.

## 2. Solution
API-first SaaS returning calibrated BBB probability, SwissADME panel, PDF report, and chat explanation.

## 3. MVP Scope
3.1 `/predict_fp` – Random-Forest BBB prob  
3.2 `/batch_predict_csv` – async job, CSV back  
3.3 `/report` – PDF v3  
3.4 Dashboard (upload & results)  
3.5 GPT-mini chat (explain)

## 4. Out-of-scope (v1)
– CYP inhibition, P-gp, payment.

## 5. Success metrics
AUC-ROC ≥0.90 external, demo > wow score 8/10.
