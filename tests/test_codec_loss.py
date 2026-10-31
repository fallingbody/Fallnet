import torch
from swarmnet.protocol.tensor_codec import TensorCodec

# Test 1: Random tensor
t_orig = torch.randn(1, 12, 576, dtype=torch.float16)
packed_bytes, scale, shape = TensorCodec.quantize_fp16_to_fp8(t_orig)
t_dequant = TensorCodec.dequantize_fp8_to_fp16(packed_bytes, device="cpu")

sim = TensorCodec.compute_cosine_similarity(t_orig, t_dequant)
print(f"Cosine Similarity for FP8 Codec: {sim:.6f}")

# Test 2: Uncompressed FP16 direct byte transfer codec for 100% loss-less baseline
header = f"FP16|{t_orig.shape[0]},{t_orig.shape[1]},{t_orig.shape[2]}|".encode()
raw_bytes = header + t_orig.cpu().numpy().tobytes()

# Decode
parts = raw_bytes.split(b"|", 2)
dims = tuple(map(int, parts[1].decode().split(",")))
t_fp16 = torch.from_numpy(np.frombuffer(parts[2], dtype=np.float16).copy()).reshape(dims)
sim_fp16 = TensorCodec.compute_cosine_similarity(t_orig, t_fp16)
print(f"Cosine Similarity for Raw FP16: {sim_fp16:.6f}")
