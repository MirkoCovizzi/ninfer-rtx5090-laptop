#include "hip/hip_runtime.h"
#pragma once

#include <hip/hip_runtime.h>

namespace ninfer::ops {

enum class Cache { ca, cg };

template <class V, class T>
__device__ __forceinline__ V load_vec(const T* ptr) {
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 ||
                  sizeof(V) == 16);
    return *reinterpret_cast<const V*>(ptr);
}

template <class V, class T>
__device__ __forceinline__ V load_ldg(const T* ptr) {
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 ||
                  sizeof(V) == 16);
    return __ldg(reinterpret_cast<const V*>(ptr));
}

template <class T, class V>
__device__ __forceinline__ void store_vec(T* ptr, V value) {
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 ||
                  sizeof(V) == 16);
    *reinterpret_cast<V*>(ptr) = value;
}

// gfx11 maps shared memory at generic base 0x1'0000'0000'0000; the LDS offset
// used by ds instructions is the low 16 bits of the generic address.
__device__ __forceinline__ unsigned smem_addr(const void* ptr) {
    return static_cast<unsigned>(reinterpret_cast<std::uintptr_t>(ptr) & 0xFFFFu);
}

template <int Bytes>
struct vec_type;
template <>
struct vec_type<4> {
    using type = unsigned;
};
template <>
struct vec_type<8> {
    using type = unsigned long long;
};
template <>
struct vec_type<16> {
    using type = struct alignas(16) {
        unsigned long long lo;
        unsigned long long hi;
    };
};

// RDNA 3.5 has no cp.async: these helpers issue a synchronous global load into
// shared memory. The commit/wait stages are no-ops; kernels that relied on the
// async pipeline still get correct data, with the loads issued in order.
template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async(void* smem_dst, const void* gmem_src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "cp_async supports 4, 8, or 16 bytes");
    (void)Policy;
    using V = typename vec_type<Bytes>::type;
    store_vec(smem_dst, load_vec<V>(gmem_src));
}

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async_zfill(void* smem_dst, const void* gmem_src,
                                               int src_bytes) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16,
                  "cp_async_zfill supports 4, 8, or 16 bytes");
    (void)Policy;
    unsigned char* dst       = static_cast<unsigned char*>(smem_dst);
    const unsigned char* src = static_cast<const unsigned char*>(gmem_src);
#pragma unroll
    for (int i = 0; i < Bytes; ++i) {
        dst[i] = i < src_bytes ? src[i] : 0;
    }
}

__device__ __forceinline__ void cp_commit() {}

template <int Groups>
__device__ __forceinline__ void cp_wait() {
    (void)Groups;
}

template <int Bytes>
__device__ __forceinline__ void pipe_copy(void* smem_dst, const void* gmem_src) {
    cp_async<Bytes>(smem_dst, gmem_src);
}

__device__ __forceinline__ void pipe_commit() {}

template <int Groups>
__device__ __forceinline__ void pipe_wait() {
    (void)Groups;
}

} // namespace ninfer::ops
