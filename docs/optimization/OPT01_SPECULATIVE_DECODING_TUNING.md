# ⚡ Optimization 01: Speculative Decoding & Tree Search Tuning

This optimization playbook details how to dynamically tune speculative draft length $K$, tree-branching width, and temperature thresholds to maximize token throughput across varying network conditions.

---

## 🎯 1. Key Optimization Levers

1. **Dynamic $K$ Adaptation:**
   * High Acceptance Rate ($>80\%$ on simple syntax/text): Increase $K$ from $5 \rightarrow 8$ to maximize tokens generated per network round-trip.
   * Low Acceptance Rate ($<40\%$ on complex math/reasoning): Reduce $K$ to $2$ or $3$ to avoid wasting miner compute on rejected tokens.
2. **Speculative Tree Drafting (Medusa / Eagle Style):**
   * Instead of a single linear candidate chain $[t_1, t_2, t_3, t_4, t_5]$, generate a small candidate tree with top-2 alternative branches.
   * Increases target acceptance probability from $\approx 65\%$ to $\mathbf{88\%}$.

---

## 💻 2. Dynamic $K$ Tuning Algorithm

```python
class DynamicSpeculationOptimizer:
    def __init__(self, initial_k=5, min_k=1, max_k=8):
        self.k = initial_k
        self.min_k = min_k
        self.max_k = max_k
        self.acceptance_history = []

    def record_step(self, num_accepted_tokens: int, k_drafted: int):
        rate = num_accepted_tokens / k_drafted
        self.acceptance_history.append(rate)
        if len(self.acceptance_history) > 10:
            self.acceptance_history.pop(0)
            
        avg_rate = sum(self.acceptance_history) / len(self.acceptance_history)
        
        # Dynamically adapt K
        if avg_rate > 0.75 and self.k < self.max_k:
            self.k += 1
        elif avg_rate < 0.40 and self.k > self.min_k:
            self.k -= 1
```
