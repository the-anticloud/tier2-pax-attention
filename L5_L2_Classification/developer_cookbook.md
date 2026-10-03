# Developer Cookbook — PAX_ATTENTION
**Stack:** Python 3.11, PyTorch 2.10+, Flash Attention 2, xformers, AIOSS_FORMAT

## Core Usage Patterns

## Initialize attention module
```python
from pax_attention import PAXAttention
attn = PAXAttention(model_config=pax_27b_config, use_flash_attn=True, kv_cache_size_mb=8192)
```

## Forward pass
```python
output, kv_cache = attn.forward(
    query=q_tensor, key=k_tensor, value=v_tensor,
    attention_mask=mask, kv_cache=prev_kv_cache
)
```

## Benchmark attention throughput
```python
result = attn.benchmark(seq_len=2048, batch_size=1, n_heads=32, head_dim=128)
print(f"Throughput: {result.tokens_per_sec:.1f} tok/s, KV cache: {result.kv_cache_mb:.0f}MB")
```

## KV cache AIOSS entry
```python
kv_hash = attn.hash_kv_cache(kv_cache)
aioss_append("./attn.aioss", kv_hash.encode(), "PAX_ATTENTION")
```

## AIOSS Append
```python
import hashlib, time

def aioss_append(chain_path, payload: bytes, module_id: str):
    entry_hash = hashlib.sha3_256(payload).digest()
    ts = int(time.time_ns()).to_bytes(8, 'big')
    with open(chain_path, 'rb') as f:
        f.seek(-32, 2); prev_hash = f.read(32)
    new_hash = hashlib.sha3_256(prev_hash + entry_hash + ts).digest()
    with open(chain_path, 'ab') as f:
        f.write(ts + entry_hash + new_hash)
    return new_hash.hex()
```

## Performance
Flash Attention 2 on T4: use `use_flash_attn=False` and xformers memory_efficient_attention as fallback. KV cache: pre-allocate for max sequence length to avoid reallocation. Grouped query attention (GQA) reduces KV cache by 4-8x for PAX 27B.

## Integration
Core dependency of PAX_INFERENCE_CORE. Optimized by PAX_CACHE (KV cache layer). Benchmarked by PAX_MONITOR.
