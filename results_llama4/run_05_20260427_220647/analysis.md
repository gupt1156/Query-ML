# Run 5 Analysis — 2026-04-27 (seed=46)

## Examples Provided to Model

### Positive Class — Good Fuel Efficiency
| Car | Attributes |
|-----|-----------|
| Car #253 | 4 cylinders, low horsepower, low displacement, light weight, fast acceleration |
| Car #75  | 8 cylinders, high horsepower, high displacement, heavy weight, fast acceleration |

### Negative Class — Bad Fuel Efficiency
| Car | Attributes |
|-----|-----------|
| Car #239 | 6 cylinders, high horsepower, medium displacement, heavy weight, fast acceleration |
| Car #78  | 4 cylinders, medium horsepower, medium displacement, average weight, fast acceleration |

> **Note:** Car #75 is a dataset outlier — it has 8-cyl, high-hp, high-disp, heavy attributes typical of bad fuel efficiency, yet is labeled as good fuel efficiency. This contradicts Car #253 on every attribute and makes the two classes nearly indistinguishable by the available features. Car #78 (negative) similarly looks like a typical good-efficiency car (4 cyl, medium hp). The poor discrimination in this run is a data sampling issue, not a model error.

---

## Generated Queries

| # | Query |
|---|-------|
| Q0 | Does the car have 8 cylinders? |
| Q1 | Does the car have high horsepower? |
| Q2 | Does the car have low displacement? |
| Q3 | Does the car have heavy weight? |
| Q4 | Does the car have 6 cylinders? |

---

## Per-Query Differentiation Analysis

### Q0 — "Does the car have 8 cylinders?"

| Car | Class | Expected | Model Answer | Correct? |
|-----|-------|----------|--------------|----------|
| Car #253 | POS | No (4 cyl) | No | ✓ |
| Car #75  | POS | Yes (8 cyl) | Yes | ✓ |
| Car #239 | NEG | No (6 cyl) | No | ✓ |
| Car #78  | NEG | No (4 cyl) | No | ✓ |

**Discrimination:** POS yes-rate = 0.5, NEG yes-rate = 0.0 → **0.5** (partial — only one pos car has 8 cyl, and that car is an outlier)

---

### Q1 — "Does the car have high horsepower?"

| Car | Class | Expected | Model Answer | Correct? |
|-----|-------|----------|--------------|----------|
| Car #253 | POS | No (low hp) | No | ✓ |
| Car #75  | POS | Yes (high hp) | Yes | ✓ |
| Car #239 | NEG | Yes (high hp) | Yes | ✓ |
| Car #78  | NEG | No (medium hp) | No | ✓ |

**Discrimination:** POS yes-rate = 0.5, NEG yes-rate = 0.5 → **0.0** (non-discriminative — both classes split 50/50 because Car #75 mirrors Car #239 on horsepower)

---

### Q2 — "Does the car have low displacement?"

| Car | Class | Expected | Model Answer | Correct? |
|-----|-------|----------|--------------|----------|
| Car #253 | POS | Yes (low disp) | Yes | ✓ |
| Car #75  | POS | No (high disp) | No | ✓ |
| Car #239 | NEG | No (medium disp) | No | ✓ |
| Car #78  | NEG | No (medium disp) | No | ✓ |

**Discrimination:** POS yes-rate = 0.5, NEG yes-rate = 0.0 → **0.5** (partial — only Car #253 has low displacement)

---

### Q3 — "Does the car have heavy weight?"

| Car | Class | Expected | Model Answer | Correct? |
|-----|-------|----------|--------------|----------|
| Car #253 | POS | No (light) | No | ✓ |
| Car #75  | POS | Yes (heavy) | Yes | ✓ |
| Car #239 | NEG | Yes (heavy) | Yes | ✓ |
| Car #78  | NEG | No (average) | No | ✓ |

**Discrimination:** POS yes-rate = 0.5, NEG yes-rate = 0.5 → **0.0** (non-discriminative — both classes split 50/50 for the same reason as Q1)

---

### Q4 — "Does the car have 6 cylinders?"

| Car | Class | Expected | Model Answer | Correct? |
|-----|-------|----------|--------------|----------|
| Car #253 | POS | No (4 cyl) | No | ✓ |
| Car #75  | POS | No (8 cyl) | No | ✓ |
| Car #239 | NEG | Yes (6 cyl) | Yes | ✓ |
| Car #78  | NEG | No (4 cyl) | No | ✓ |

**Discrimination:** POS yes-rate = 0.0, NEG yes-rate = 0.5 → **0.5** (partial — only one neg car has 6 cyl)

---

## Summary

| Query | Discrimination | Notes |
|-------|---------------|-------|
| Q0 — 8 cylinders? | 0.5 | Partial: only Car #75 (POS outlier) has 8 cyl |
| Q1 — High horsepower? | 0.0 | Non-discriminative: POS and NEG both split 50/50 |
| Q2 — Low displacement? | 0.5 | Partial: only Car #253 has low disp |
| Q3 — Heavy weight? | 0.0 | Non-discriminative: POS and NEG both split 50/50 |
| Q4 — 6 cylinders? | 0.5 | Partial: only Car #239 has 6 cyl |
| **Average** | **0.3** | |

**Individual answer accuracy: 20/20 (100%)**

## Conclusion

The model answered all 20 individual queries correctly. However, the **average discrimination is low at 0.3** — not because of model errors, but because the sampled examples make the two classes nearly inseparable:

- **Car #75** (positive, good efficiency) has attributes — 8 cylinders, high horsepower, high displacement, heavy weight — that are identical to a stereotypical bad-efficiency car and directly contradict its positive-class label.
- **Car #78** (negative, bad efficiency) has attributes — 4 cylinders, medium horsepower, average weight, fast acceleration — that resemble a typical good-efficiency car.

This sampling collision means no query based on the available features can cleanly separate the two classes. Queries Q1 (high horsepower) and Q3 (heavy weight) achieve zero discrimination because Car #75 and Car #239 are attribute mirrors of each other across the class boundary. The queries generated are sensible given what the model observed; the issue lies in the random sample drawn for this run.
