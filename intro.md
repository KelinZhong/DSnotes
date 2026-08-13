# Data Science Interview Prep

Personal study guides, cheatsheets, and runnable references covering the full data science interview loop — from the pandas coding round to product-sense and shipping decisions.

Written for people with a **statistics or quantitative background** moving into data science roles: the assumption throughout is that you already have the maths, and what these notes add is the interview framing, the product context, and the engineering practice around it.

To read other notes, visit [Study Notes](https://kelinzhong.github.io/StudyNotes/).

## Contents

- **Python Data Library & Data Manipulation** — pandas/NumPy references and data manipulation drills (basic → advanced)
- **Business Metrics** — The 12 core product/business metrics (DAU/MAU, retention, K-factor, NPS, conversion, LTV, CAC, MRR, CTR, GMV, and more), each implemented in both pandas and SQL with matching section numbering
- **Machine Learning** — How models learn, evaluation, data preparation, unsupervised learning, tree-based models, similarity/boundary methods, and class imbalance, calibration & decision thresholds
- **Experiment** — A/B test design, analysis, and methods beyond A/B testing
- **Causal Inference and Machine Learning** — Foundations, experiments, observational methods, identification, CATE/meta-learners, uplift modeling, causal forests, and an end-to-end case
- **Data Science Workflow** — Problem framing through EDA, features, modeling, validation, deployment, and communication, with analytics and ML case studies
- **Business Decisions** — Evaluating model results, cost-benefit analysis, and the shipping decision

## Suggested reading order

The chapters follow the arc of a typical DS interview loop: coding round (Data Libs) → metric definitions (Metrics) → ML concepts → A/B testing (Experiment) → causal methods (Causal ML) → case studies (DS Workflow) → product sense (Business Decisions).

If you already have a statistics background, the sampling and statistical-inference references are largely review — skim them and spend the time on Experiment, Data Science Workflow, and Business Decisions instead.

## Notebook template

Most content notebooks follow the same structure, so any of them can be navigated the same way:

> **What it is → Key intuition → Assumptions → Validation (with thresholds) → If violated → Formula → Reference → Interview Q&A → Gotchas**

Every notebook is executed when the book is built, so a green build means every code cell here actually runs.
