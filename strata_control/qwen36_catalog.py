"""Shared Qwen3.6 identity data; importing this module installs nothing.

The installer and local-file preview use the same official file names, sizes and
hash declarations. A preview never establishes a payload hash or engine fit.
"""

ARCHITECTURE = "qwen35moe"
FILE_TEMPLATE = "Qwen3.6-35B-A3B-{q}.gguf"
FILES = {
    "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf": (22853663008, "55983c5a75a1ab969824077b3bb3de4146e82a9234072b48ad4e8f92ad3fe9f1"),
    "Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf": (32611711264, "35fce994cd36104a7dc1bd8a4bdf13778145664c00fdef6773aebc9246e5019c"),
    "Qwen3.6-35B-A3B-Q8_0.gguf": (37801097504, "c1283d8b80c3e38b2735ddbc9766d3b3126f44d6c484be419d4e101d09a76131"),
    "mmproj-F16.gguf": (899283584, "71f3cbc1f7cc0f30d09d41cfa924c0060827ebc33bf15ace7e86661e856f0160"),
}
MODELS = {"UD-Q4_K_XL": {"about": "4-bit (Unsloth Dynamic), the smallest", "ram_gb": 32},
          "UD-Q6_K_XL": {"about": "6-bit (Unsloth Dynamic), closer to the full model", "ram_gb": 44},
          "Q8_0": {"about": "8-bit, the closest to the full model", "ram_gb": 52}}

# Aggregate union from the engine's gate/up and down encoding support. Actual
# support also depends on tensor role, shape and the installed backend/build.
# Q6_K in particular requires STRATA_Q6K_EXPERTS. Do not infer runtime support
# for every role from membership here, or a quantization solely from its name.
# src/kernels/cuda/iq_kernels.cu:843-853,5459-5464 at release v0.2.8.
EXPERT_TYPES = frozenset({"IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ3_S", "IQ2_S", "IQ4_XS",
                          "IQ1_M", "Q2_0", "Q4_K", "Q5_K", "Q5_0", "Q4_0", "Q4_1",
                          "Q8_0", "NVFP4", "TQ2_T", "TQK6", "TQP5", "TQP6", "TQP7",
                          "IQ4_NL", "Q5_1", "TQK7", "Q6_K"})
