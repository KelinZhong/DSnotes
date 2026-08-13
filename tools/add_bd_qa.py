#!/usr/bin/env python3
"""Append Interview Q&A blocks to the Business Decisions chapter.

The audit found "Interview Q&A" present in every Machine Learning, Causal ML
and Data Science Workflow content notebook, but absent from all of Business
Decisions and all of Experiment. Business Decisions is treated first because it
is the thinnest chapter (3 content notebooks, ~35 KB, zero code cells) and
covers precisely the product-sense material a statistics-background candidate
is least likely to have rehearsed.

Purely additive: appends one cell per notebook, touching nothing existing.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BD = ROOT / "Business Decisions"

QA: dict[str, str] = {}

QA["bd1_evaluating_results.ipynb"] = """
---
## Interview Q&A

**Q: An A/B test shows a 2% lift with p = 0.04. Do you ship?**
A: Not on that alone. I'd check the sequence first: was there a Sample Ratio
Mismatch, was the primary metric pre-registered, and how many metrics were
tested? A p of 0.04 across ten metrics is roughly what noise produces. Then I'd
ask whether 2% clears the *practical* bar, not just the statistical one — the
confidence interval matters more than the point estimate here, because if it
runs from 0.1% to 3.9% the plausible outcomes include "not worth the
engineering cost."

**Q: Offline AUC improved from 0.78 to 0.83, but the A/B test was flat. What
happened?**
A: Several standard explanations, and I'd work through them in order. Offline
evaluation measures ranking on historical data; the A/B test measures behaviour
under a live policy. The gap usually comes from (a) the offline test set
leaking, (b) the model improving on a segment that rarely gets traffic, (c) the
decision threshold not being re-tuned so better scores never became different
actions, or (d) the metric the model optimises not driving the business metric
the test measures. I'd check (c) first — it is the most common and the cheapest
to fix.

**Q: A stakeholder says "the model is 92% accurate, let's launch."**
A: I'd ask what the base rate is, because on an imbalanced problem that number
may be indistinguishable from doing nothing. Then I'd redirect to the operating
point: at the threshold we would actually deploy, how many true positives do we
catch and how many false positives does the ops team absorb? That reframes the
conversation from a model property to a staffing and cost question, which is
what the launch decision actually turns on.

**Q: Your observational causal estimate says the feature drove +8% retention.
How much do you trust it?**
A: It depends entirely on which assumption is doing the work. I'd state the
identification strategy, then give a robustness number rather than a point
estimate alone — Rosenbaum bounds, an E-value, or Oster's δ, depending on the
method. If the estimate flips under a plausible unmeasured confounder, I'd
present it as directional evidence and push for an A/B test before committing
budget.

**Q: When is weaker evidence good enough?**
A: When the decision is cheap to reverse. A fully reversible change with a
small blast radius does not need experimental-grade evidence; an irreversible
one — a pricing change, a migration, anything customer-visible and sticky —
does. I'd match the strength of evidence to the cost of being wrong rather than
applying one bar to everything.

### Gotchas
- "Statistically significant" and "worth shipping" are different questions; a large sample makes trivial effects significant
- Segment findings discovered *after* seeing the results are hypotheses, not conclusions
- A flat A/B test is evidence of no *detectable* effect at your power, not evidence of no effect
"""

QA["bd2_cost_benefit.ipynb"] = """
---
## Interview Q&A

**Q: Walk me through sizing the business value of a churn model.**
A: I'd build it as a chain and be explicit about the weakest link. Start with
the population at risk per period, multiply by the model's precision at the
deployment threshold to get true positives, then by the intervention's success
rate — which is a *causal* quantity and usually the shakiest input — then by
the margin retained per save. Subtract intervention cost across everyone
contacted, including false positives. I'd present a range driven by the success
rate rather than a single number, because that is where the uncertainty
concentrates.

**Q: What is the most common error in these calculations?**
A: Treating the model's identified population as the model's incremental
impact. Some of the customers flagged as at-risk would have stayed regardless,
and the intervention only creates value for the ones whose behaviour it
changes. That is an uplift question, not a prediction question — the honest
version needs a holdout, or an uplift model.

**Q: How do you handle a break-even that depends on an unknown input?**
A: Invert the question. Rather than guessing the input, solve for the value at
which the project breaks even and ask whether that value is plausible. "This
pays for itself if the intervention works 4% of the time" is a much easier
conversation than defending a made-up 12%, and it hands the domain expert a
question they can actually answer.

**Q: The model is worth $2M a year. Should we build it?**
A: Only relative to the alternative. The relevant comparison is not against
doing nothing, it is against the next-best use of the same team-quarters — and
against the simple heuristic baseline, which frequently captures a large share
of the value at a fraction of the cost. If a rules-based version gets 70% of
the benefit in two weeks, the ML project has to justify the remaining 30%.

**Q: How do you account for ongoing cost?**
A: I separate one-time build cost from the run rate: inference, monitoring,
retraining, on-call, and the human review capacity the threshold implies. The
review cost is the one people forget, and it scales with false positives — so
the threshold choice and the cost model are the same decision, not two.

### Gotchas
- Annualising a lift measured over two weeks assumes no novelty decay and no seasonality; both are usually false
- Benefits are typically estimated with uncertainty and costs stated as points, which biases every business case optimistic
- A cost-benefit case built on uncalibrated probabilities is arithmetically fine and substantively wrong
"""

QA["bd3_shipping_decision.ipynb"] = """
---
## Interview Q&A

**Q: The results are ambiguous and the deadline is tomorrow. What do you do?**
A: I'd separate what is reversible from what is not. If the change can be
rolled back cheaply and the guardrails held, I'd ship to a small ring with
auto-rollback triggers defined in advance — that converts a decision into an
experiment. If it is irreversible, ambiguity is a reason not to ship, and I'd
say so explicitly along with what evidence would change my answer and how long
it would take to get.

**Q: Guardrail metrics moved slightly negative but the primary metric won.
Ship?**
A: I'd ask whether the guardrail movement is within the range we pre-committed
to tolerate. Guardrails exist precisely so this is decided before we see the
data — relitigating the threshold afterwards is how organisations talk
themselves into shipping regressions. If it was not pre-specified, I'd treat
the movement as a real risk, size it in the same units as the win, and let the
comparison decide.

**Q: How do you decide between a 1% ramp and a 50/50 launch?**
A: By blast radius, not by confidence. The ramp exists to bound the damage from
the failure mode I have not thought of, so the question is what a bad outcome
costs during the exposure window. High-traffic, easily-monitored, reversible
changes can ramp fast; anything touching payments, notifications, or data
retention ramps slowly regardless of how good the test looked.

**Q: What is the cost of not shipping?**
A: Real and routinely ignored. Delay has an opportunity cost — the foregone
lift for every week you wait — plus the compounding cost of the team's time and
the option value of learning from production. "Wait for more data" is a
decision with a price tag, and it should be quoted alongside the risk it
avoids.

**Q: The model shipped and metrics degraded three months later. What went
wrong?**
A: I'd distinguish drift from decay in the decision layer. Feature or label
drift shows up in input distributions and PSI; a stale threshold shows up as
stable model scores with a shifting base rate. Both are monitoring failures
rather than modelling failures, which is why the launch checklist has to
include what is monitored, what triggers retraining, and who is paged.

### Gotchas
- Shipping is a decision under uncertainty, not a proof; demanding certainty produces a team that ships nothing
- Deciding the rollback trigger after launch means you will not pull it
- "We'll monitor it" is not a plan unless a named person owns a named dashboard with a named threshold
"""


def main() -> None:
    for fname, text in QA.items():
        path = BD / fname
        nb = json.loads(path.read_text(encoding="utf-8"))
        if any("## Interview Q&A" in "".join(c["source"]) for c in nb["cells"]):
            print(f"  skip {fname}: already has Interview Q&A")
            continue
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True),
        })
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"  {fname}: +1 Interview Q&A cell ({len(nb['cells'])} cells total)")


if __name__ == "__main__":
    main()
