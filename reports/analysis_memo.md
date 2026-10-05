# Memo — To Ritu Deshpande, Head of D2C Operations

## Decision
Put the model into a **review queue**, not an automatic rejection gate. Rank claims by fraud score and send the highest-risk claims to the investigation desk, capped at the desk's 40 claims/month.

## What the validation says
On a time-based June 2026 holdout, the model reached **97.26% accuracy** and **0.892 ROC-AUC**. The top 40 ranked claims contained **6 frauds (15%)**, with ₹7,734 of observed fraudulent claim value in that holdout, or **₹193 per reviewed claim**.

This is a validation estimate, not a promise for the hidden test set.

## Why the model is policy-aware
The operating policy changed on 1 May 2026: claims below ₹2,000 became auto-approved without inspection. The model is developed and validated inside that newer regime instead of treating the prior 13 months as identical.

## New partners
New-partner tenure is included, but I would **not** label all newer partners as fraudulent. The score combines partner tenure with claim amount, inspection status, customer history, warranty timing and other claim signals.

## What I would do next week
1. Use the ranked queue for the 40 investigations available each month.
2. Track precision and rupees prevented separately from board-level accuracy.
3. Review the top partner patterns with Service Desk before changing partner policy.
4. Re-estimate the model monthly as investigation outcomes accumulate.
5. Do not auto-deny a claim from the model alone; retain human investigation for the high-risk queue.

## Caveat
Fraud is rare and only 37 labeled frauds fall in the post-May development regime. The model is therefore best treated as a prioritisation tool until more post-change outcomes accumulate.
