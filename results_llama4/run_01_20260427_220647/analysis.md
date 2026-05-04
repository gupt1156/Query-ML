# Run 1 Analysis — 2026-04-27 (seed=42)

## Examples Provided to Model

### Positive Class — Good Fuel Efficiency
| Car | Attributes |
|-----|-----------|
| Car #281 | 4 cylinders, low horsepower, low displacement, light weight, fast acceleration |
| Car #233 | 4 cylinders, low horsepower, medium displacement, average weight, fast acceleration |

### Negative Class — Bad Fuel Efficiency
| Car | Attributes |
|-----|-----------|
| Car #277 | 8 cylinders, high horsepower, high displacement, heavy weight, slow acceleration |
| Car #214 | 6 cylinders, high horsepower, medium displacement, heavy weight, moderate acceleration |

---

## Generated Queries

| # | Query |
|---|-------|
| Q0 | Does the car have 8 cylinders? |
| Q1 | Does the car have high horsepower? |
| Q2 | Does the car have heavy weight? |
| Q3 | Does the car have slow acceleration? |
| Q4 | Does the car have 4 cylinders? |

---

## Per-Query Differentiation Analysis

### Q0 — "Does the car have 8 cylinders?"

| Car | Class | Expected | Model Answer | Correct? |
|-----|-------|----------|--------------|----------|
| Car #281 | POS | No (4 cyl) | No | ✓ |
| Car #233 | POS | No (4 cyl) | No | ✓ |
| Car #277 | NEG | Yes (8 cyl) | Yes | ✓ |
| Car #214 | NEG | No (6 cyl) | No | ✓ |

**Discrimination:** POS yes-rate = 0.0, NEG yes-rate = 0.5 → **0.5** (partial — only one neg car has 8 cyl)

---

### Q1 — "Does the car have high horsepower?"

| Car | Class | Expected | Model Answer | Correct? |
|-----|-------|----------|--------------|----------|
| Car #281 | POS | No (low hp) | No | ✓ |
| Car #233 | POS | No (low hp) | No | ✓ |
| Car #277 | NEG | Yes (high hp) | Yes | ✓ |
| Car #214 | NEG | Yes (high hp) | Yes | ✓ |

**Discrimination:** POS yes-rate = 0.0, NEG yes-rate = 1.0 → **1.0** (perfect)

---

### Q2 — "Does the car have heavy weight?"

| Car | Class | Expected | Model Answer | Correct? |
|-----|-------|----------|--------------|----------|
| Car #281 | POS | No (light) | No | ✓ |
| Car #233 | POS | No (average) | No | ✓ |
| Car #277 | NEG | Yes (heavy) | Yes | ✓ |
| Car #214 | NEG | Yes (heavy) | Yes | ✓ |

**Discrimination:** POS yes-rate = 0.0, NEG yes-rate = 1.0 → **1.0** (perfect)

---

### Q3 — "Does the car have slow acceleration?"

| Car | Class | Expected | Model Answer | Correct? |
|-----|-------|----------|--------------|----------|
| Car #281 | POS | No (fast) | No | ✓ |
| Car #233 | POS | No (fast) | No | ✓ |
| Car #277 | NEG | Yes (slow) | Yes | ✓ |
| Car #214 | NEG | No (moderate) | No | ✓ |

**Discrimination:** POS yes-rate = 0.0, NEG yes-rate = 0.5 → **0.5** (partial — Car #214 has moderate, not slow, acceleration)

---

### Q4 — "Does the car have 4 cylinders?"

| Car | Class | Expected | Model Answer | Correct? |
|-----|-------|----------|--------------|----------|
| Car #281 | POS | Yes (4 cyl) | Yes | ✓ |
| Car #233 | POS | Yes (4 cyl) | Yes | ✓ |
| Car #277 | NEG | No (8 cyl) | No | ✓ |
| Car #214 | NEG | No (6 cyl) | No | ✓ |

**Discrimination:** POS yes-rate = 1.0, NEG yes-rate = 0.0 → **1.0** (perfect)

---

## Summary

| Query | Discrimination | Notes |
|-------|---------------|-------|
| Q0 — 8 cylinders? | 0.5 | Partial: one neg has 6 cyl |
| Q1 — High horsepower? | 1.0 | Perfect |
| Q2 — Heavy weight? | 1.0 | Perfect |
| Q3 — Slow acceleration? | 0.5 | Partial: one neg has moderate acc |
| Q4 — 4 cylinders? | 1.0 | Perfect |
| **Average** | **0.8** | |

**Individual answer accuracy: 20/20 (100%)**

## Conclusion

The model accurately answered all queries based on the provided car descriptions. Three of the five queries perfectly discriminated between the two classes (discrimination = 1.0). The two partial queries (Q0 and Q3) reflect genuine within-class variation in the negative examples — Car #214 has 6 cylinders and moderate acceleration rather than 8 cylinders and slow acceleration — not model errors. Overall discrimination is strong at **0.8**.
