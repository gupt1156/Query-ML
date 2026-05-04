# Run 3 Analysis — 2026-04-27 (seed=44)

## Examples Provided to Model

### Positive Class — Good Fuel Efficiency
| Car | Attributes |
|-----|-----------|
| Car #95  | 4 cylinders, low horsepower, low displacement, light weight, fast acceleration |
| Car #358 | 4 cylinders, low horsepower, low displacement, light weight, fast acceleration |

### Negative Class — Bad Fuel Efficiency
| Car | Attributes |
|-----|-----------|
| Car #89  | 8 cylinders, high horsepower, high displacement, heavy weight, moderate acceleration |
| Car #344 | 6 cylinders, high horsepower, high displacement, heavy weight, moderate acceleration |

---

## Generated Queries

| # | Query |
|---|-------|
| Q0 | Does the car have 8 cylinders? |
| Q1 | Does the car have high horsepower? |
| Q2 | Does the car have low displacement? |
| Q3 | Does the car have heavy weight? |
| Q4 | Does the car have 4 cylinders? |

---

## Per-Query Differentiation Analysis

### Q0 — "Does the car have 8 cylinders?"

| Car | Class | Expected | Model Answer | Correct? |
|-----|-------|----------|--------------|----------|
| Car #95  | POS | No (4 cyl) | No | ✓ |
| Car #358 | POS | No (4 cyl) | No | ✓ |
| Car #89  | NEG | Yes (8 cyl) | Yes | ✓ |
| Car #344 | NEG | No (6 cyl) | No | ✓ |

**Discrimination:** POS yes-rate = 0.0, NEG yes-rate = 0.5 → **0.5** (partial — Car #344 has 6 cylinders)

---

### Q1 — "Does the car have high horsepower?"

| Car | Class | Expected | Model Answer | Correct? |
|-----|-------|----------|--------------|----------|
| Car #95  | POS | No (low hp) | No | ✓ |
| Car #358 | POS | No (low hp) | No | ✓ |
| Car #89  | NEG | Yes (high hp) | Yes | ✓ |
| Car #344 | NEG | Yes (high hp) | Yes | ✓ |

**Discrimination:** POS yes-rate = 0.0, NEG yes-rate = 1.0 → **1.0** (perfect)

---

### Q2 — "Does the car have low displacement?"

| Car | Class | Expected | Model Answer | Correct? |
|-----|-------|----------|--------------|----------|
| Car #95  | POS | Yes (low disp) | Yes | ✓ |
| Car #358 | POS | Yes (low disp) | Yes | ✓ |
| Car #89  | NEG | No (high disp) | No | ✓ |
| Car #344 | NEG | No (high disp) | No | ✓ |

**Discrimination:** POS yes-rate = 1.0, NEG yes-rate = 0.0 → **1.0** (perfect)

---

### Q3 — "Does the car have heavy weight?"

| Car | Class | Expected | Model Answer | Correct? |
|-----|-------|----------|--------------|----------|
| Car #95  | POS | No (light) | No | ✓ |
| Car #358 | POS | No (light) | No | ✓ |
| Car #89  | NEG | Yes (heavy) | Yes | ✓ |
| Car #344 | NEG | Yes (heavy) | Yes | ✓ |

**Discrimination:** POS yes-rate = 0.0, NEG yes-rate = 1.0 → **1.0** (perfect)

---

### Q4 — "Does the car have 4 cylinders?"

| Car | Class | Expected | Model Answer | Correct? |
|-----|-------|----------|--------------|----------|
| Car #95  | POS | Yes (4 cyl) | Yes | ✓ |
| Car #358 | POS | Yes (4 cyl) | Yes | ✓ |
| Car #89  | NEG | No (8 cyl) | No | ✓ |
| Car #344 | NEG | No (6 cyl) | No | ✓ |

**Discrimination:** POS yes-rate = 1.0, NEG yes-rate = 0.0 → **1.0** (perfect)

---

## Summary

| Query | Discrimination | Notes |
|-------|---------------|-------|
| Q0 — 8 cylinders? | 0.5 | Partial: Car #344 has 6 cyl |
| Q1 — High horsepower? | 1.0 | Perfect |
| Q2 — Low displacement? | 1.0 | Perfect |
| Q3 — Heavy weight? | 1.0 | Perfect |
| Q4 — 4 cylinders? | 1.0 | Perfect |
| **Average** | **0.9** | |

**Individual answer accuracy: 20/20 (100%)**

## Conclusion

The model answered all queries correctly. The two positive examples were essentially identical, providing very clean and consistent signal. Four of five queries achieved perfect discrimination. The partial score on Q0 is expected: Car #344 has 6 cylinders, not 8, so "8 cylinders?" correctly returns No for that car. The model recognised this nuance accurately. Overall discrimination is excellent at **0.9**.
