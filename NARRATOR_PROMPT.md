# Experiment Narrator — prompt

Turns a Difference-in-Differences experiment readout into a short, data-driven
executive conclusion. Used in the app's **Results** tab (`experiment.narrate_did`),
with a deterministic fallback so it always produces output; set an API key to use a
real LLM. Generic — swap the payload for any experiment's metrics.

## Prompt

```text
You are a senior data scientist writing the conclusion of a ride-hailing
dynamic-pricing experiment. Given the Difference-in-Differences results below,
write <=120 words that: (1) state the GMV and Trip lift with significance,
(2) name the key trade-off, (3) give ONE ship/iterate recommendation with a
guardrail to watch, (4) flag one next experiment. Be specific, no preamble.

EXPERIMENT (JSON):
{payload}
```

`{payload}` is the experiment result as JSON, e.g.:

```json
{
  "design": "A/B",
  "treatment": "Elasticity-LP (cap 1.15)",
  "control": "Flat (no surge)",
  "gmv":   {"lift_pct": 18.26, "ci": [17.56, 18.92], "p_value": 0.0},
  "trips": {"lift_pct": 4.44,  "ci": [4.02, 4.81],   "p_value": 0.0}
}
```

## Example output (deterministic fallback)

> In an A/B test of **Elasticity-LP (cap 1.15)** vs **Flat (no surge)**, DiD estimates a
> **GMV lift of +18.3%** (95% CI [+17.6, +18.9]) and a **Trip lift of +4.4%**
> (95% CI [+4.0, +4.8]) — significant. GMV and trips both rise; the trade-off to watch
> is average surge vs rider price sensitivity. **Recommendation: Ship the treatment and
> ramp gradually**, with 95th-percentile rider wait and average surge as guardrails.
> Next experiment: re-run as a switchback per zone-cluster and test an online/streaming
> surge update that re-solves within the day rather than a daily batch.

## Why it works

- **Role, objective and required elements are fixed up front** — the model reasons about
  the trade-off and a decision, not just restating numbers.
- **Numbered, testable requirements** (lift + significance, trade-off, recommendation +
  guardrail, next experiment) keep the output structured and comparable across runs.
- **Metrics passed as JSON**, so the same prompt drops onto any experiment readout.
