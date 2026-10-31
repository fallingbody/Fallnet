try:
    import psutil
except ImportError:
    psutil = None
import torch
import time
import math

class AutoProfiler:
    """
    Hardware-Aware Auto-Profiler.
    Benchmarks the host machine's RAM and compute capabilities at startup
    to dynamically determine how many transformer layers it can host.
    """
    
    def __init__(self, model_total_layers: int = 30, memory_per_layer_mb: float = 18.0):
        self.model_total_layers = model_total_layers
        self.memory_per_layer_mb = memory_per_layer_mb
        # Reserve 200MB for OS and PyTorch overhead
        self.reserved_overhead_mb = 200.0 

    def get_available_memory_mb(self) -> float:
        """Returns the available system memory in Megabytes."""
        if psutil is not None:
            try:
                mem = psutil.virtual_memory()
                return mem.available / (1024 * 1024)
            except Exception:
                pass
        return 2048.0 # Safe default 2GB fallback

    def benchmark_flops(self) -> float:
        """Runs a tiny FP16 matrix multiplication to estimate device speed."""
        try:
            # 1000x1000 matrix
            a = torch.randn(1000, 1000, dtype=torch.float32)
            b = torch.randn(1000, 1000, dtype=torch.float32)
            
            start = time.time()
            for _ in range(10):
                _ = torch.matmul(a, b)
            end = time.time()
            
            elapsed = end - start
            # roughly 2e9 operations per matmul, 10 iterations -> 20e9 ops
            flops = 20e9 / elapsed
            return flops
        except Exception:
            return 1e9 # Fallback 1 GFLOP

    def calculate_layer_capacity(self) -> int:
        """
        Dynamically calculates how many layers this node can host 
        based on available memory and a baseline compute requirement.
        """
        available_mb = self.get_available_memory_mb()
        usable_mb = max(0, available_mb - self.reserved_overhead_mb)
        
        # Max layers based on RAM
        ram_layer_capacity = math.floor(usable_mb / self.memory_per_layer_mb)
        
        # Optional: scale down if compute is extremely slow
        flops = self.benchmark_flops()
        compute_multiplier = 1.0
        if flops < 5e9: # very slow device (e.g. old phone)
            compute_multiplier = 0.5 
            
        final_capacity = int(ram_layer_capacity * compute_multiplier)
        
        # Cap at total model layers
        return min(max(1, final_capacity), self.model_total_layers)

    def profile(self) -> dict:
        capacity = self.calculate_layer_capacity()
        return {
            "available_ram_mb": round(self.get_available_memory_mb(), 2),
            "estimated_gflops": round(self.benchmark_flops() / 1e9, 2),
            "recommended_layer_capacity": capacity
        }

if __name__ == "__main__":
    profiler = AutoProfiler()
    print("🖥️  Running Auto-Profiler...")
    stats = profiler.profile()
    print(f"📊 Available RAM: {stats['available_ram_mb']} MB")
    print(f"⚡ Compute Speed: {stats['estimated_gflops']} GFLOPS")
    print(f"✅ Recommended Capacity: {stats['recommended_layer_capacity']} Layers")
