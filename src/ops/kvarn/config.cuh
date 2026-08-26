#pragma once

// Fixed native profile from Huawei KVarN commit 7586257f1c632e63187bfacbbe21ccb51540f7b3,
// vllm/model_executor/layers/quantization/kvarn/config.py: kvarn_k4v2_g64, D256.

namespace ninfer::ops::kvarn {

inline constexpr int D          = 256;
inline constexpr int Group      = 64;
inline constexpr int KBits      = 4;
inline constexpr int VBits      = 2;
inline constexpr int Iterations = 8;
inline constexpr int PrefillSlabTokens = 16384;

inline constexpr float StdMin = 1.0e-3F;
inline constexpr float StdMax = 1.0e3F;
inline constexpr float LogMin = -0.3F;
inline constexpr float LogMax = 10.0F;

} // namespace ninfer::ops::kvarn
