# Run 2 Analysis — 2026-04-27 (seed=43)

## Examples Provided to Model

### Positive Class — Good Fuel Efficiency
| Car | Attributes |
|-----|-----------|
| Car #113 | 4 cylinders, low horsepower, low displacement, light weight, moderate acceleration |
| Car #287 | 4 cylinders, medium horsepower, low displacement, average weight, fast acceleration |

### Negative Class — Bad Fuel Efficiency
| Car | Attributes |
|-----|-----------|
| Car #115 | 8 cylinders, high horsepower, high displacement, heavy weight, slow acceleration |
| Car #282 | 8 cylinders, high horsepower, high displacement, heavy weight, slow acceleration |

---

## Generated Queries

| # | Query |
|---|-------|
| Q0 | Does the car have 8 cylinders? |
| Q1 | Does the car have high horsepower? |
| Q2 | Does the car have high displacement? |
| Q3 | Does the car have heavy weight? |
| Q4 | Does the car have fast acceleration? |

---

## Per-Query Differentiation Analysis

### Q0 — "Does the car have 8 cylinders?"

| Car | Class | Expected | Model Answer | Correct? |
|-----|-------|----------|--------------|----------|
| Car #113 | POS | No (4 cyl) | No | ✓ |
| Car #287 | POS | No (4 cyl) | No | ✓ |
| Car #115 | NEG | Yes (8 cyl) | Yes | ✓ |
| Car #282 | NEG | Yes (8 cyl) | Yes | ✓ |

**Discrimination:** POS yes-rate = 0.0, NEG yes-rate = 1.0 → **1.0** (perfect)

---

### Q1 — "Does the car have high horsepower?"

| Car | Class | Expected | Model Answer | Correct? |
|-----|-------|----------|--------------|----------|
| Car #113 | POS | No (low hp) | No | ✓ |
| Car #287 | POS | No (medium hp) | No | ✓ |
| Car #115 | NEG | Yes (high hp) | Yes | ✓ |
| Car #282 | NEG | Yes (high hp) | Yes | ✓ |

**Discrimination:** POS yes-rate = 0.0, NEG yes-rate = 1.0 → **1.0** (perfect; model correctly treated medium hp as "not high")

---

### Q2 — "Does the car have high displacement?"

| Car | Class | Expected | Model Answer | Correct? |
|-----|-------|----------|--------------|----------|
| Car #113 | POS | No (low disp) | No | ✓ |
| Car #287 | POS | No (low disp) | No | ✓ |
| Car #115 | NEG | Yes (high disp) | Yes | ✓ |
| Car #282 | NEG | Yes (high disp) | Yes | ✓ |

**Discrimination:** POS yes-rate = 0.0, NEG yes-rate = 1.0 → **1.0** (perfect)

---

### Q3 — "Does the car have heavy weight?"

| Car | Class | Expected | Model Answer | Correct? |
|-----|-------|----------|--------------|----------|
| Car #113 | POS | No (light) | No | ✓ |
| Car #287 | POS | No (average) | No | ✓ |
| Car #115 | NEG | Yes (heavy) | Yes | ✓ |
| Car #282 | NEG | Yes (heavy) | Yes | ✓ |

**Discrimination:** POS yes-rate = 0.0, NEG yes-rate = 1.0 → **1.0** (perfect; model correctly treated average weight as "not heavy")

---

### Q4 — "Does the car have fast acceleration?"

| Car | Class | Expected | Model Answer | Correct? |
|-----|-------|----------|--------------|----------|
| Car #113 | POS | No (moderate) | No | ✓ |
| Car #287 | POS | Yes (fast) | Yes | ✓ |
| Car #115 | NEG | No (slow) | No | ✓ |
| Car #282 | NEG | No (slow) | No | ✓ |

**Discrimination:** POS yes-rate = 0.5, NEG yes-rate = 0.0 → **0.5** (partial — Car #113 has moderate, not fast, acceleration)

---

## Summary

| Query | Discrimination | Notes |
|-------|---------------|-------|
| Q0 — 8 cylinders? | 1.0 | Perfect |
| Q1 — High horsepower? | 1.0 | Perfect |
| Q2 — High displacement? | 1.0 | Perfect |
| Q3 — Heavy weight? | 1.0 | Perfect |
| Q4 — Fast acceleration? | 0.5 | Partial: Car #113 has moderate acc |
| **Average** | **0.9** | |

**Individual answer accuracy: 20/20 (100%)**

## Conclusion

The model answered all queries correctly. Four of five queries achieved perfect discrimination (1.0), with only Q4 being partial because Car #113 has moderate rather than fast acceleration. The two negative examples were nearly identical (both 8-cyl, high-hp, high-disp, heavy, slow-acc), giving the model very clean signal. Overall discrimination is excellent at **0.9**.
