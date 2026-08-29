#include "ops/attn_input_proj/bf16/bf16_attn_input_plan.h"

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {

void bf16_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                              Tensor& k, Tensor& v, cudaStream_t stream) {
    if (x.ne[1] <= kBf16AttnInputSmallTDispatchEnd) {
        for (std::int32_t begin = 0; begin < x.ne[1]; begin += kBf16AttnInputSmallTMaxTokens) {
            const std::int32_t count = std::min(kBf16AttnInputSmallTMaxTokens, x.ne[1] - begin);
            Tensor x_chunk           = x.slice(1, begin, count);
            Tensor q_chunk           = q.slice(1, begin, count);
            Tensor gate_chunk        = gate.slice(1, begin, count);
            Tensor k_chunk           = k.slice(1, begin, count);
            Tensor v_chunk           = v.slice(1, begin, count);
            bf16_attn_input_small_t_launch(x_chunk, weight, q_chunk, gate_chunk, k_chunk, v_chunk,
                                           stream);
        }
        return;
    }
    bf16_attn_input_mma_launch(x, weight, q, gate, k, v, stream);
}

} // namespace ninfer::ops::detail
