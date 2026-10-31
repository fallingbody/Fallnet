# 🧪 Test Suite 05: Benchmark, TTFT & Latency Profiling

This test suite measures key performance indicators (KPIs) of the AirCompute mesh across diverse prompt lengths, batch sizes, and network conditions.

---

## 🎯 1. Key Performance Indicators (KPIs)

* **Time-to-First-Token (TTFT):** Target $<350\text{ms}$ for standard prompt prefills.
* **Inter-Token Latency (ITL):** Target $<30\text{ms}$ per token with speculative decoding ($K=5$).
* **Effective Throughput:** Target $>35\text{ tokens/sec}$ over residential broadband ($>25\text{ Mbps}$).
* **Network Bandwidth per Token:** Target $<12\text{ KB}$ per verified token block.

---

## 💻 2. Automated Benchmark Profiler

```python
import time
import statistics

def benchmark_swarm_generation(client, test_prompts: list):
    """Measures TTFT, ITL, and throughput across sample prompts"""
    ttft_list, itl_list, throughput_list = [], [], []
    
    for prompt in test_prompts:
        start_time = time.perf_counter()
        token_times = []
        
        token_stream = client.stream_generate(prompt, max_tokens=100)
        for token in token_stream:
            token_times.append(time.perf_counter())
            
        ttft = (token_times[0] - start_time) * 1000.0
        itls = [(token_times[i] - token_times[i-1]) * 1000.0 for i in range(1, len(token_times))]
        total_time = token_times[-1] - start_time
        throughput = len(token_times) / total_time
        
        ttft_list.append(ttft)
        itl_list.extend(itls)
        throughput_list.append(throughput)
        
    print(f"📊 BENCHMARK RESULTS:")
    print(f"  • Median TTFT: {statistics.median(ttft_list):.2f} ms")
    print(f"  • P95 ITL: {statistics.quantiles(itl_list, n=20)[18]:.2f} ms")
    print(f"  • Avg Throughput: {statistics.mean(throughput_list):.2f} tokens/sec")
```
