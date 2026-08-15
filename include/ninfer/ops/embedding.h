#pragma once

#include "core/tensor.h"

#include <hip/hip_runtime.h>

namespace ninfer::ops {

/**
 * Gathers one embedding row per token:
 *
 *   ideal[d,t] = dequantize(table)[ids[t],d].
 *
 * `ids` is contiguous I32 [T], `out` is contiguous BF16 [D,T], and every id is in
 * [0,vocab). `table` has logical shape [vocab,D] and is contiguous BF16_CTRL, Q6G64_F16S
 * RowSplit, or W8G32_F16S RowSplit with FP16 scales. Dense BF16 values are copied bit-exactly. For
 * quantized tables, the oracle independently decodes each signed code and multiplies it by the
 * exact stored FP16 scale in FP64; the BF16 output is promoted and compared directly with that
 * ideal. Final output storage rounding belongs to the quantized embedding criterion, not the
 * oracle. The registered domains are Q6/D=5120 and W8/D=2048 or D=5120. `out` must not overlap
 * `ids` or any table plane. There is no workspace or persistent state side effect.
 */
void embedding(const Tensor& ids, const Weight& table, Tensor& out, hipStream_t stream);

} // namespace ninfer::ops
