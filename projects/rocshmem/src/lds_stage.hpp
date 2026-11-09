/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

// LDS-staged async IPC copy: global<->global byte copies staged through LDS,
// double-buffered and tiled. Both gfx1250 only, picked by USE_TDM:
//   - TDM (requires USE_TDM): tensor_load_to_lds/store_from_lds,
//     2D-descriptor tensor-mover engine, VGPR-free both legs.
//   - async (USE_LDS_STAGE without USE_TDM): sized (b32) global<->LDS VMEM
//     instructions, independent of TDM, VGPR-free both legs.

#ifndef LIBRARY_SRC_LDS_STAGE_HPP_
#define LIBRARY_SRC_LDS_STAGE_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

// USE_TDM implies USE_LDS_STAGE.
#if defined(USE_TDM) && !defined(USE_LDS_STAGE)
#define USE_LDS_STAGE
#endif

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

#if defined(__HIP_PLATFORM_AMD__) &&                       \
    __has_builtin(__builtin_amdgcn_tensor_load_to_lds) &&   \
    __has_builtin(__builtin_amdgcn_tensor_store_from_lds)
#define HIP_HAVE_TDM_INTRINSICS 1
#else
#define HIP_HAVE_TDM_INTRINSICS 0
#endif

#if defined(__HIP_PLATFORM_AMD__) &&                                   \
    __has_builtin(__builtin_amdgcn_global_load_async_to_lds_b32) &&    \
    __has_builtin(__builtin_amdgcn_global_store_async_from_lds_b32) && \
    __has_builtin(__builtin_amdgcn_asyncmark) &&                      \
    __has_builtin(__builtin_amdgcn_wait_asyncmark)
#define HIP_HAVE_LDS_ASYNC_INTRINSICS 1
#else
#define HIP_HAVE_LDS_ASYNC_INTRINSICS 0
#endif

// Both backends are gfx1250-only; USE_TDM picks TDM, plain USE_LDS_STAGE picks async.
#if defined(USE_TDM) && (defined(__gfx1250__) || defined(__gfx1250_strict__)) && HIP_HAVE_TDM_INTRINSICS
#define ROCSHMEM_LDS_STAGE_ENABLED 1
#define ROCSHMEM_LDS_STAGE_BACKEND_TDM 1
#elif defined(USE_LDS_STAGE) && !defined(USE_TDM) && HIP_HAVE_LDS_ASYNC_INTRINSICS
#define ROCSHMEM_LDS_STAGE_ENABLED 1
#define ROCSHMEM_LDS_STAGE_BACKEND_ASYNC 1
#else
#define ROCSHMEM_LDS_STAGE_ENABLED 0
#endif

namespace rocshmem {
namespace lds_stage {

// Under plain USE_LDS_STAGE, not ROCSHMEM_LDS_STAGE_ENABLED: host code needs
// this too, and host compiles don't have the arch macros ENABLED depends on.
#if defined(USE_LDS_STAGE)
constexpr uint32_t FlatCopyElementLog2 = 3;  // 8 bytes/element

__host__ __device__ constexpr size_t lds_bytes_for_tile(uint32_t tile_bytes) {
  return 2 * static_cast<size_t>(tile_bytes);
}
#endif  // USE_LDS_STAGE

#if ROCSHMEM_LDS_STAGE_ENABLED

// Largest multiple of tile_bytes_cap not exceeding total_bytes (or a single
// undersized tile if total_bytes is smaller than one).
__device__ __forceinline__ size_t covered_bytes(size_t total_bytes, uint32_t tile_bytes_cap) {
  if (total_bytes == 0) return 0;
  if (total_bytes < tile_bytes_cap) {
    constexpr size_t elem_bytes = size_t{1} << FlatCopyElementLog2;
    return total_bytes - (total_bytes % elem_bytes);
  }
  return (total_bytes / tile_bytes_cap) * static_cast<size_t>(tile_bytes_cap);
}

#if defined(ROCSHMEM_LDS_STAGE_BACKEND_TDM)
// TDM backend: wraps __builtin_amdgcn_tensor_load_to_lds/store_from_lds.
namespace tdm {

using u32x4 = uint32_t __attribute__((ext_vector_type(4)));
using i32x4 = int32_t __attribute__((ext_vector_type(4)));
using i32x8 = int32_t __attribute__((ext_vector_type(8)));

// GROUP1 data_size field.
enum class DataSize : uint32_t { Bits8 = 0, Bits16 = 1, Bits32 = 2, Bits64 = 3 };

// cachepolicy = th_bits[2:0] | (scope_bits[1:0] << 3).
// Memory scope (cachepolicy bits[4:3]), pre-shifted into place here.
enum class Scope : unsigned {
  Wgp = 0,           // workgroup processor: L1-level coherence (default)
  ShaderEngine = 8,  // GL2/L2-level coherence within one shader engine
  Device = 16,       // all shader engines on this GPU
  System = 24,       // CPU + all GPUs on the node
};

// Temporal hint (cachepolicy bits[2:0]) for tensor_load_to_lds. th=7 is
// undefined for loads, so it's intentionally not a member here.
enum class LoadTemporalHint : unsigned {
  Default = 0,                 // write-allocate, cached in L1 and L2
  NonTemporal = 1,             // TH_LOAD_NT: no L1 allocation
  HwManaged = 2,               // TH_LOAD_HT: hardware-managed temporal hint
  LastUse = 3,                 // TH_LOAD_LU: evict from L1 after this read
  NonTemporalReadThrough = 4,  // TH_LOAD_NT_RT
  ReadThroughNonTemporal = 5,  // TH_LOAD_RT_NT
  NonTemporalHwManaged = 6,    // TH_LOAD_NT_HT
};

// Temporal hint for tensor_store_from_lds. Same bit positions as
// LoadTemporalHint, but th=3/th=7 mean something different for stores.
enum class StoreTemporalHint : unsigned {
  Default = 0,                 // write-back
  NonTemporal = 1,             // TH_STORE_NT: non-temporal write-through
  HwManaged = 2,               // TH_STORE_HT: hardware-managed temporal hint
  WriteBack = 3,               // TH_STORE_WB: explicit write-back
  NonTemporalReadThrough = 4,  // TH_STORE_NT_RT
  ReadThroughNonTemporal = 5,  // TH_STORE_RT_NT
  NonTemporalHwManaged = 6,    // TH_STORE_NT_HT
  NonTemporalWriteBack = 7,    // TH_STORE_NT_WB
};

constexpr int make_cache_policy(LoadTemporalHint th, Scope scope = Scope::Wgp) {
  return static_cast<int>(th) | static_cast<int>(scope);
}
constexpr int make_cache_policy(StoreTemporalHint th, Scope scope = Scope::Wgp) {
  return static_cast<int>(th) | static_cast<int>(scope);
}

// DESCRIPTOR CONSTRUCTION
// GROUP0 (addresses + control) and GROUP1 (tensor layout) for a 2D transfer.
// GROUP2/GROUP3 (3D/gather) aren't modeled; every call site here is 2D.
struct Descriptor2D {
  u32x4 group0{};
  i32x8 group1{};
};

// GROUP0 only (LDS address + 48-bit global VA), split out so callers can
// cheaply rebuild just the addresses of an otherwise-invariant descriptor.
__device__ __forceinline__ u32x4 make_group0(uintptr_t lds_addr, uintptr_t global_addr) {
  return u32x4{
      1u,  // m_count = 1
      static_cast<uint32_t>(lds_addr),
      static_cast<uint32_t>(global_addr & 0xFFFFFFFFu),
      // bits[24:0] = global_addr_hi, bits[31:30] = m_type = 2 (non-gather).
      // m_th is left 0: it's overridden by the intrinsic's cachepolicy
      // argument, and m_is_store is set automatically by the compiler.
      (2u << 30) | static_cast<uint32_t>((global_addr >> 32) & 0x1FFFFFFu),
  };
}

// GROUP1: element size, full-tensor/tile dims, and strides (in elements).
// Strides are genuinely 48-bit here, unlike AMD's own amd_gfx1250_TDM.h
// header, whose tensorDimNStride() setters truncate to 32 bits.
__device__ __forceinline__ i32x8 make_group1(DataSize data_size, uint32_t tensor_dim0,
                                             uint32_t tensor_dim1, uint32_t tile_dim0,
                                             uint32_t tile_dim1, uint64_t tensor_dim0_stride,
                                             uint64_t tensor_dim1_stride,
                                             uint32_t workgroup_mask = 1) {
  uint32_t s[8]{};
  s[0] = (workgroup_mask & 0xFFFFu) | (static_cast<uint32_t>(data_size) << 16);
  s[1] = (tensor_dim0 & 0xFFFFu) << 16;  // atomic_barrier_address left 0
  s[2] = ((tensor_dim0 >> 16) & 0xFFFFu) | ((tensor_dim1 & 0xFFFFu) << 16);
  s[3] = ((tensor_dim1 >> 16) & 0xFFFFu) | ((tile_dim0 & 0xFFFFu) << 16);
  s[4] = (tile_dim1 & 0xFFFFu);  // tile_dim2 left 0 (2D only)
  s[5] = static_cast<uint32_t>(tensor_dim0_stride & 0xFFFFFFFFull);
  s[6] = static_cast<uint32_t>((tensor_dim0_stride >> 32) & 0xFFFFull) |
         (static_cast<uint32_t>(tensor_dim1_stride & 0xFFFFull) << 16);
  s[7] = static_cast<uint32_t>((tensor_dim1_stride >> 16) & 0xFFFFFFFFull);

  i32x8 g1{};
  for (int i = 0; i < 8; ++i) g1[i] = static_cast<int32_t>(s[i]);
  return g1;
}

__device__ __forceinline__ Descriptor2D make_descriptor_2d(
    uintptr_t lds_addr, uintptr_t global_addr, DataSize data_size, uint32_t tensor_dim0,
    uint32_t tensor_dim1, uint32_t tile_dim0, uint32_t tile_dim1, uint64_t tensor_dim0_stride,
    uint64_t tensor_dim1_stride, uint32_t workgroup_mask = 1) {
  Descriptor2D d;
  d.group0 = make_group0(lds_addr, global_addr);
  d.group1 = make_group1(data_size, tensor_dim0, tensor_dim1, tile_dim0, tile_dim1,
                         tensor_dim0_stride, tensor_dim1_stride, workgroup_mask);
  return d;
}

// Cnt/CachePolicy are template params: the builtins require an immarg.
template <uint16_t Cnt>
__device__ __forceinline__ void tensor_wait() {
  __builtin_amdgcn_s_wait_tensorcnt(Cnt);
}

template <int CachePolicy>
__device__ __forceinline__ void load_to_lds(u32x4 group0, i32x8 group1,
                                            i32x4 group2 = i32x4{0, 0, 0, 0},
                                            i32x4 group3 = i32x4{0, 0, 0, 0}) {
  const i32x8 group4{0, 0, 0, 0, 0, 0, 0, 0};  // reserved; always zero
  __builtin_amdgcn_tensor_load_to_lds(group0, group1, group2, group3, group4, CachePolicy);
}

template <int CachePolicy>
__device__ __forceinline__ void store_from_lds(u32x4 group0, i32x8 group1,
                                               i32x4 group2 = i32x4{0, 0, 0, 0},
                                               i32x4 group3 = i32x4{0, 0, 0, 0}) {
  const i32x8 group4{0, 0, 0, 0, 0, 0, 0, 0};  // reserved; always zero
  __builtin_amdgcn_tensor_store_from_lds(group0, group1, group2, group3, group4, CachePolicy);
}

template <int CachePolicy>
__device__ __forceinline__ void load_to_lds(const Descriptor2D &d) {
  load_to_lds<CachePolicy>(d.group0, d.group1);
}

template <int CachePolicy>
__device__ __forceinline__ void store_from_lds(const Descriptor2D &d) {
  store_from_lds<CachePolicy>(d.group0, d.group1);
}

}  // namespace tdm
#endif  // ROCSHMEM_LDS_STAGE_BACKEND_TDM

#if defined(ROCSHMEM_LDS_STAGE_BACKEND_ASYNC)
// Async backend (gfx1250 only): sized global_load/store_async_*_lds_b32
// VMEM instructions, both legs VGPR-free. asyncmark()/wait_asyncmark(N)
// lower to s_wait_asynccnt, one counter draining both op kinds -- a wait
// compatible with TDM's tensor_wait<Cnt>() pipeline shapes.
namespace async_lds {

// 4 bytes/issue; the builtins support larger sizes too, unused here.
constexpr uint32_t ElementBytes = 4;

// aux: bit0=sc0, bit1=sc1, bit4=scc, bit31=compiler-implemented volatile.
constexpr int32_t LoadAux = 0;
constexpr int32_t StoreAux = 0;

__device__ __forceinline__ void issue_load_chunk(void *lds_addr, const void *global_addr) {
  __builtin_amdgcn_global_load_async_to_lds_b32(
      reinterpret_cast<int *>(const_cast<void *>(global_addr)),
      reinterpret_cast<int *>(lds_addr), 0, LoadAux);
}

__device__ __forceinline__ void issue_store_chunk(void *global_addr, const void *lds_addr) {
  __builtin_amdgcn_global_store_async_from_lds_b32(
      reinterpret_cast<int *>(global_addr),
      reinterpret_cast<int *>(const_cast<void *>(lds_addr)), 0, StoreAux);
}

__device__ __forceinline__ void mark() { __builtin_amdgcn_asyncmark(); }

template <uint16_t Cnt>
__device__ __forceinline__ void wait_mark() {
  __builtin_amdgcn_wait_asyncmark(Cnt);
}

// This lane's strided slice of tile_bytes, one mark() per tile.
__device__ __forceinline__ void issue_load_tile_lane(void *lds_tile, const void *src_tile,
                                                     uint32_t tile_bytes, uint32_t lane_id,
                                                     uint32_t num_lanes) {
  auto *lds_bytes = static_cast<uint8_t *>(lds_tile);
  const auto *src_bytes = static_cast<const uint8_t *>(src_tile);
  for (uint32_t off = lane_id * ElementBytes; off < tile_bytes; off += num_lanes * ElementBytes) {
    issue_load_chunk(lds_bytes + off, src_bytes + off);
  }
  mark();
}

// Store leg, mirrors issue_load_tile_lane.
__device__ __forceinline__ void issue_store_tile_lane(void *dst_tile, const void *lds_tile,
                                                      uint32_t tile_bytes, uint32_t lane_id,
                                                      uint32_t num_lanes) {
  auto *dst_bytes = static_cast<uint8_t *>(dst_tile);
  const auto *lds_bytes = static_cast<const uint8_t *>(lds_tile);
  for (uint32_t off = lane_id * ElementBytes; off < tile_bytes; off += num_lanes * ElementBytes) {
    issue_store_chunk(dst_bytes + off, lds_bytes + off);
  }
  mark();
}

}  // namespace async_lds
#endif  // ROCSHMEM_LDS_STAGE_BACKEND_ASYNC

// Unified surface copy_pipelined_shallow/deep call; TDM ignores lane_id/
// num_lanes (single-lane issue), async stripes both legs across them.
template <int LoadCachePolicy, int StoreCachePolicy>
__device__ __forceinline__ void issue_load_tile(void *lds_tile, const void *src_tile,
                                                uint32_t tile_bytes,
                                                [[maybe_unused]] uint32_t lane_id,
                                                [[maybe_unused]] uint32_t num_lanes) {
#if defined(ROCSHMEM_LDS_STAGE_BACKEND_TDM)
  const uint32_t tile_dim0 = tile_bytes >> FlatCopyElementLog2;
  tdm::Descriptor2D desc = tdm::make_descriptor_2d(
      reinterpret_cast<uintptr_t>(lds_tile), reinterpret_cast<uintptr_t>(src_tile),
      tdm::DataSize::Bits64, /*tensor_dim0=*/tile_dim0, /*tensor_dim1=*/1,
      /*tile_dim0=*/tile_dim0, /*tile_dim1=*/1, /*tensor_dim0_stride=*/tile_dim0,
      /*tensor_dim1_stride=*/tile_dim0);
  tdm::load_to_lds<LoadCachePolicy>(desc);
#elif defined(ROCSHMEM_LDS_STAGE_BACKEND_ASYNC)
  async_lds::issue_load_tile_lane(lds_tile, src_tile, tile_bytes, lane_id, num_lanes);
#endif
}

template <int LoadCachePolicy, int StoreCachePolicy>
__device__ __forceinline__ void issue_store_tile(void *dst_tile, const void *lds_tile,
                                                 uint32_t tile_bytes,
                                                 [[maybe_unused]] uint32_t lane_id,
                                                 [[maybe_unused]] uint32_t num_lanes) {
#if defined(ROCSHMEM_LDS_STAGE_BACKEND_TDM)
  const uint32_t tile_dim0 = tile_bytes >> FlatCopyElementLog2;
  tdm::Descriptor2D desc = tdm::make_descriptor_2d(
      reinterpret_cast<uintptr_t>(lds_tile), reinterpret_cast<uintptr_t>(dst_tile),
      tdm::DataSize::Bits64, /*tensor_dim0=*/tile_dim0, /*tensor_dim1=*/1,
      /*tile_dim0=*/tile_dim0, /*tile_dim1=*/1, /*tensor_dim0_stride=*/tile_dim0,
      /*tensor_dim1_stride=*/tile_dim0);
  tdm::store_from_lds<StoreCachePolicy>(desc);
#elif defined(ROCSHMEM_LDS_STAGE_BACKEND_ASYNC)
  async_lds::issue_store_tile_lane(dst_tile, lds_tile, tile_bytes, lane_id, num_lanes);
#endif
}

// wait<0>() drains all in-flight ops; wait<Cnt>() leaves up to Cnt.
template <uint16_t Cnt>
__device__ __forceinline__ void wait() {
#if defined(ROCSHMEM_LDS_STAGE_BACKEND_TDM)
  tdm::tensor_wait<Cnt>();
#elif defined(ROCSHMEM_LDS_STAGE_BACKEND_ASYNC)
  async_lds::wait_mark<Cnt>();
#endif
}

// Tile-by-tile copy through lds_scratch. double_buffered: pipeline depth 1
// (load N+1 overlaps store N, drained every pair). Otherwise serialized.
template <int LoadCachePolicy, int StoreCachePolicy>
__device__ inline void copy_pipelined_shallow(void *dst, const void *src, size_t bytes,
                                              void *lds_scratch, uint32_t tile_bytes_cap,
                                              bool double_buffered, uint32_t lane_id = 0,
                                              uint32_t num_lanes = 1) {
  const size_t covered = covered_bytes(bytes, tile_bytes_cap);
  if (covered == 0) return;
  const uint32_t tile_bytes =
      (bytes < tile_bytes_cap) ? static_cast<uint32_t>(covered) : tile_bytes_cap;
  const size_t full_tiles = covered / tile_bytes;

  auto *dst_bytes = static_cast<char *>(dst);
  const auto *src_bytes = static_cast<const char *>(src);
  auto *lds_base = static_cast<uint8_t *>(lds_scratch);

  if (double_buffered) {
    uintptr_t ping = reinterpret_cast<uintptr_t>(lds_base);
    uintptr_t pong = reinterpret_cast<uintptr_t>(lds_base + tile_bytes);

    // Prologue: start loading tile 0 into ping.
    issue_load_tile<LoadCachePolicy, StoreCachePolicy>(reinterpret_cast<void *>(ping), src_bytes,
                                                       tile_bytes, lane_id, num_lanes);

    for (size_t i = 0; i < full_tiles; ++i) {
      const bool has_next = (i + 1) < full_tiles;

      // Wait for the current tile's load to land before storing from it.
      wait<0>();

      // Prefetch tile i+1 into pong and store tile i from ping — both
      // enqueued before the next wait, so loads/stores can overlap.
      if (has_next) {
        issue_load_tile<LoadCachePolicy, StoreCachePolicy>(
            reinterpret_cast<void *>(pong), src_bytes + (i + 1) * tile_bytes, tile_bytes, lane_id,
            num_lanes);
      }
      issue_store_tile<LoadCachePolicy, StoreCachePolicy>(
          dst_bytes + i * tile_bytes, reinterpret_cast<void *>(ping), tile_bytes, lane_id,
          num_lanes);

      uintptr_t tmp = ping;
      ping = pong;
      pong = tmp;
    }
    // Drain the last in-flight load+store pair.
    wait<0>();
  } else {
    const uintptr_t lds_addr = reinterpret_cast<uintptr_t>(lds_base);
    for (size_t i = 0; i < full_tiles; ++i) {
      const size_t offset = i * tile_bytes;

      issue_load_tile<LoadCachePolicy, StoreCachePolicy>(
          reinterpret_cast<void *>(lds_addr), src_bytes + offset, tile_bytes, lane_id, num_lanes);
      wait<0>();

      issue_store_tile<LoadCachePolicy, StoreCachePolicy>(
          dst_bytes + offset, reinterpret_cast<void *>(lds_addr), tile_bytes, lane_id, num_lanes);
      wait<0>();
    }
  }
}

// Same as copy_pipelined_shallow, but pipeline depth 2 (wait<1>() instead of
// draining every iteration): op N-2 is always done by the time op N waits.
template <int LoadCachePolicy, int StoreCachePolicy>
__device__ inline void copy_pipelined_deep(void *dst, const void *src, size_t bytes,
                                           void *lds_scratch, uint32_t tile_bytes_cap,
                                           uint32_t lane_id = 0, uint32_t num_lanes = 1) {
  const size_t covered = covered_bytes(bytes, tile_bytes_cap);
  if (covered == 0) return;
  const uint32_t tile_bytes =
      (bytes < tile_bytes_cap) ? static_cast<uint32_t>(covered) : tile_bytes_cap;
  const size_t full_tiles = covered / tile_bytes;

  auto *dst_bytes = static_cast<char *>(dst);
  const auto *src_bytes = static_cast<const char *>(src);
  const uintptr_t buf[2] = {
      reinterpret_cast<uintptr_t>(lds_scratch),
      reinterpret_cast<uintptr_t>(static_cast<uint8_t *>(lds_scratch) + tile_bytes)};

  auto issue_load = [&](uintptr_t lds_addr, const void *global_addr) {
    issue_load_tile<LoadCachePolicy, StoreCachePolicy>(reinterpret_cast<void *>(lds_addr),
                                                       global_addr, tile_bytes, lane_id,
                                                       num_lanes);
  };
  auto issue_store = [&](uintptr_t lds_addr, void *global_addr) {
    issue_store_tile<LoadCachePolicy, StoreCachePolicy>(
        global_addr, reinterpret_cast<void *>(lds_addr), tile_bytes, lane_id, num_lanes);
  };

  // Prologue: load tile 0 into buf[0].
  issue_load(buf[0], src_bytes);

  for (size_t tile_even = 0; tile_even < full_tiles; tile_even += 2) {
    const size_t tile_odd = tile_even + 1;
    const bool has_odd = tile_odd < full_tiles;

    if (has_odd) {
      wait<1>();
      issue_load(buf[tile_odd & 1u], src_bytes + tile_odd * tile_bytes);
    }

    // No odd tile: load(tile_even) was only one op ago, not two, so
    // wait<1>() would race ahead of it -- drain fully instead.
    if (has_odd) {
      wait<1>();
    } else {
      wait<0>();
    }
    issue_store(buf[tile_even & 1u], dst_bytes + tile_even * tile_bytes);

    if (has_odd) {
      wait<1>();
      issue_store(buf[tile_odd & 1u], dst_bytes + tile_odd * tile_bytes);

      const size_t next_even = tile_even + 2;
      if (next_even < full_tiles) {
        wait<1>();
        issue_load(buf[next_even & 1u], src_bytes + next_even * tile_bytes);
      }
    }
  }

  // Drain everything still in flight before returning.
  wait<0>();
}

// Backs rocshmem_set_lds_stage()/rocshmem_query_lds_stage_bytes(). Every
// thread must call set_lds() (barrier needs all of them) before any put/get
// in that kernel using this path; wave put/get splits the buffer per wave.
struct LdsRegistration {
  void* ptr;
  size_t bytes;
};

inline __shared__ LdsRegistration lds_registration;

__device__ __forceinline__ void set_lds(void* ptr, size_t bytes) {
  if (hipThreadIdx_x == 0 && hipThreadIdx_y == 0 && hipThreadIdx_z == 0) {
    lds_registration = LdsRegistration{ptr, bytes};
  }
  __builtin_amdgcn_s_barrier();  // publish to the rest of the block
}

__device__ __forceinline__ LdsRegistration get_lds() {
  return lds_registration;
}

#endif  // ROCSHMEM_LDS_STAGE_ENABLED

}  // namespace lds_stage
}  // namespace rocshmem

#endif  // LIBRARY_SRC_LDS_STAGE_HPP_
