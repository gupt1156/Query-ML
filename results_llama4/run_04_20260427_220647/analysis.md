# Run 4 Analysis — 2026-04-27 (seed=45)

## Examples Provided to Model

### Positive Class — Good Fuel Efficiency
| Car | Attributes |
|-----|-----------|
| Car #93  | 4 cylinders, low horsepower, low displacement, light weight, fast acceleration |
| Car #152 | 4 cylinders, low horsepower, low displacement, light weight, moderate acceleration |

### Negative Class — Bad Fuel Efficiency
| Car | Attributes |
|-----|-----------|
| Car #87  | 8 cylinders, high horsepower, high displacement, heavy weight, slow acceleration |
| Car #140 | 8 cylinders, high horsepower, high displacement, heavy weight, moderate acceleration |

---

## Generated Queries

| # | Query |
|---|-------|
| Q0 | Does the car have 8 cylinders? |
| Q1 | Does the car have high horsepower? |
| Q2 | Does the car have high displacement? |
| Q3 | Does the car have heavy weight? |
| Q4 | Does the car have slow acceleration? |

---

## Per-Query Differentiation Analysis

### Q0 — "Does the car have 8 cylinders?"

| Car | Class | Expected | Model Answer | Correct? |
|-----|-------|----------|--------------|----------|
| Car #93  | POS | No (4 cyl) | No | ✓ |
| Car #152 | POS | No (4 cyl) | No | ✓ |
| Car #87  | NEG | Yes (8 cyl) | Yes | ✓ |
| Car #140 | NEG | Yes (8 cyl) | Yes | ✓ |

**Discrimination:** POS yes-rate = 0.0, NEG yes-rate = 1.0 → **1.0** (perfect)

---

### Q1 — "Does the car have high horsepower?"

| Car | Class | Expected | Model Answer | Correct? |
|-----|-------|----------|--------------|----------|
| Car #93  | POS | No (low hp) | No | ✓ |
| Car #152 | POS | No (low hp) | No | ✓ |
| Car #87  | NEG | Yes (high hp) | Yes | ✓ |
| Car #140 | NEG | Yes (high hp) | Yes | ✓ |

**Discrimination:** POS yes-rate = 0.0, NEG yes-rate = 1.0 → **1.0** (perfect)

---

### Q2 — "Does the car have high displacement?"

| Car | Class | Expected | Model Answer | Correct? |
|-----|-------|----------|--------------|----------|
| Car #93  | POS | No (low disp) | No | ✓ |
| Car #152 | POS | No (low disp) | No | ✓ |
| Car #87  | NEG | Yes (high disp) | Yes | ✓ |
| Car #140 | NEG | Yes (high disp) | Yes | ✓ |

**Discrimination:** POS yes-rate = 0.0, NEG yes-rate = 1.0 → **1.0** (perfect)

---

### Q3 — "Does the car have heavy weight?"

| Car | Class | Expected | Model Answer | Correct? |
|-----|-------|----------|--------------|----------|
| Car #93  | POS | No (light) | No | ✓ |
| Car #152 | POS | No (light) | No | ✓ |
| Car #87  | NEG | Yes (heavy) | Yes | ✓ |
| Car #140 | NEG | Yes (heavy) | Yes | ✓ |

**Discrimination:** POS yes-rate = 0.0, NEG yes-rate = 1.0 → **1.0** (perfect)

---

### Q4 — "Does the car have slow acceleration?"

| Car | Class | Expected | Model Answer | Correct? |
|-----|-------|----------|--------------|----------|
| Car #93  | POS | No (fast) | No | ✓ |
| Car #152 | POS | No (moderate) | No | ✓ |
| Car #87  | NEG | Yes (slow) | Yes | ✓ |
| Car #140 | NEG | No (moderate) | No | ✓ |

**Discrimination:** POS yes-rate = 0.0, NEG yes-rate = 0.5 → **0.5** (partial — Car #140 has moderate, not slow, acceleration)

---

## Summary

| Query | Discrimination | Notes |
|-------|---------------|-------|
| Q0 — 8 cylinders? | 1.0 | Perfect |
| Q1 — High horsepower? | 1.0 | Perfect |
| Q2 — High displacement? | 1.0 | Perfect |
| Q3 — Heavy weight? | 1.0 | Perfect |
| Q4 — Slow acceleration? | 0.5 | Partial: Car #140 has moderate acc |
| **Average** | **0.9** | |

**Individual answer accuracy: 20/20 (100%)**

## Conclusion

The model answered all queries correctly. The two classes were very cleanly separated across four attributes (cylinders, horsepower, displacement, weight). Q4 achieves partial discrimination because Car #140 has moderate rather than slow acceleration — this is accurately reflected in the model's answer. Overall discrimination is excellent at **0.9**, limited only by within-class variation in acceleration that was already present in the data.
