#pragma once

// Self-contained replacement for the hipcub::BlockMergeSort usage in the sampling
// kernels (ROCm is not guaranteed to ship hipCUB headers).
//
// Sorts BLOCK_THREADS * ITEMS_PER_THREAD keys across the block with an exact
// total order given by the caller's comparator, leaving thread t's items as the
// keys of ranks [t * ITEMS_PER_THREAD, (t + 1) * ITEMS_PER_THREAD) — the same
// thread-major distribution hipcub::BlockMergeSort::Sort produces.
//
// Requires BLOCK_THREADS * ITEMS_PER_THREAD to be a power of two.

#include <hip/hip_runtime.h>

namespace ninfer::ops {

template <class T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
struct BlockMergeSortShim {
    static_assert((BLOCK_THREADS * ITEMS_PER_THREAD & (BLOCK_THREADS * ITEMS_PER_THREAD - 1)) == 0,
                  "BlockMergeSortShim requires a power-of-two key count");

    struct TempStorage {
        alignas(16) T keys[BLOCK_THREADS * ITEMS_PER_THREAD];
    };

    __device__ __forceinline__ explicit BlockMergeSortShim(TempStorage& storage) : storage_(storage) {}

    // Sorts the block's keys descending by the comparator (comp(a, b) means
    // "a is ordered before b"). Bitonic merge network; every step is an exact
    // comparison, so the result is deterministic and ties are resolved by the
    // caller's total order.
    template <class Comp>
    __device__ __forceinline__ void Sort(T (&keys)[ITEMS_PER_THREAD], Comp comp) {
        constexpr int N = BLOCK_THREADS * ITEMS_PER_THREAD;
        const int tid  = static_cast<int>(threadIdx.x);
#pragma unroll
        for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
            storage_.keys[tid * ITEMS_PER_THREAD + item] = keys[item];
        }
        __syncthreads();
        for (int k = 2; k <= N; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                const int i    = tid;
                const int ixj  = i ^ j;
                if (ixj > i) {
                    T a = storage_.keys[i];
                    T b = storage_.keys[ixj];
                    if ((i & k) == 0) {
                        // descending merge: first half keeps the larger element
                        if (comp(b, a)) {
                            storage_.keys[i]    = b;
                            storage_.keys[ixj] = a;
                        }
                    } else {
                        // second half keeps the smaller element
                        if (comp(a, b)) {
                            storage_.keys[i]    = b;
                            storage_.keys[ixj] = a;
                        }
                    }
                }
                __syncthreads();
            }
        }
#pragma unroll
        for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
            keys[item] = storage_.keys[tid * ITEMS_PER_THREAD + item];
        }
        __syncthreads();
    }

private:
    TempStorage& storage_;
};

} // namespace ninfer::ops
