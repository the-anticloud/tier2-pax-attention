# Deploy Guide — PAX_ATTENTION
**Platform:** Anticloud PAX 27B harness | Air-gap capable

## Prerequisites
Python 3.11+, PyTorch 2.10+, flash-attn 2.5+, xformers 0.0.24+, CUDA 12.x

## Environment
T4 GPU minimum (15.6GB VRAM). CUDA 12.x. 32GB RAM for full KV cache. Flash Attention 2 requires sm80+ (A100) for maximum throughput; T4 (sm75) uses compatibility mode.

## AIOSS Integration
```bash
aioss init --module PAX_ATTENTION --output ./pax_attention.aioss
aioss append --chain ./pax_attention.aioss --payload ./output.bin --module PAX_ATTENTION
aioss verify --chain ./pax_attention.aioss
```

## Air-Gap Deployment
```bash
pip download -r requirements.txt -d ./wheels/
# Transfer to air-gap host
pip install --no-index --find-links ./wheels/ -r requirements.txt
```

## PAX 27B Harness Wiring
```python
from anticloud_pax import PAXHarness
harness = PAXHarness(model_path="./pax-27b-q4.gguf", module="PAX_ATTENTION",
                     aioss_chain="./pax_attention.aioss",
                     classification="L5_NARROW_L2_GENERAL")
result = harness.process(input_data)
```

## Verification
```bash
aioss verify --chain ./pax_attention.aioss --verbose
python -m pax_attention.tests.smoke
```
