// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP16/FP32 narrowing to FP8/BF8 with nearest or seeded stochastic rounding.
// Owns public entry points and their validation/dispatch policy.

#include "common/float_formats.h"
#include "common/internal.h"
#include "conversion/fp8_narrow_impl.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

using rocjitsu::arith::Fp8NarrowPolicy;

// Narrow FP16/FP32 to FP8/BF8 for the selected lanes. Reject invalid controls before changing
// outputs.
template <bool Bf8, bool Stochastic, bool Half = false>
int convert(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
            const uint32_t *const *a, const uint32_t *const *b,
            Fp8NarrowPolicy policy = Fp8NarrowPolicy::Gfx12) {
  if (mode >> 32) {
    return rocjitsu::arith::execute_dpp(
        flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
          return convert<Bf8, Stochastic, Half>(flags, exec_mask, uint32_t(mode), d, source, b,
                                                policy);
        });
  }
  const uint32_t known =
      RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_A |
      (Stochastic ? RJ_ARITH_CVT_BYTE_3
                  : RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_HIGH_D) |
      (Half ? RJ_ARITH_ALU_HIGH_A : 0);
  if (int error = rocjitsu::arith::validate(flags, mode & ~known)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  bool saturate = !Half && (flags & RJ_ARITH_FP16_OVFL);
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    rocjitsu::arith::fp8_narrow_x86_64_v4<Bf8, Stochastic, Half>(exec_mask, mode, saturate, d[0],
                                                                 a[0], b ? b[0] : nullptr, policy);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    rocjitsu::arith::fp8_narrow_x86_64_v3<Bf8, Stochastic, Half>(exec_mask, mode, saturate, d[0],
                                                                 a[0], b ? b[0] : nullptr, policy);
    return RJ_ARITH_SUCCESS;
  }
#endif
  unsigned shift = Stochastic ? ((mode >> 16) & 3) * 8 : mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  uint32_t selected = (Stochastic ? 255u : 65535u) << shift;
  if (!Stochastic && shift == 0 &&
      (policy == Fp8NarrowPolicy::Fnuz || policy == Fp8NarrowPolicy::Ocp)) {
    selected = UINT32_MAX;
  }
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    uint32_t av = a[0][lane], bv = b ? b[0][lane] : 0;
    if constexpr (Half) {
      if constexpr (!Stochastic) {
        bv = rocjitsu::arith::as_bits(rocjitsu::arith::f16_to_float(uint16_t(av >> 16)));
      }
      int shift = Stochastic && (mode & RJ_ARITH_ALU_HIGH_A) ? 16 : 0;
      av = rocjitsu::arith::as_bits(rocjitsu::arith::f16_to_float(uint16_t(av >> shift)));
    }
    av &= mode & RJ_ARITH_ALU_ABS_A ? 0x7fffffff : UINT32_MAX;
    av ^= mode & RJ_ARITH_ALU_NEG_A ? 0x80000000 : 0;
    uint32_t value = rocjitsu::arith::narrow_fp8<Bf8, Stochastic>(av, bv, saturate, policy);
    if constexpr (!Stochastic) {
      bv &= mode & RJ_ARITH_ALU_ABS_B ? 0x7fffffff : UINT32_MAX;
      bv ^= mode & RJ_ARITH_ALU_NEG_B ? 0x80000000 : 0;
      value |= rocjitsu::arith::narrow_fp8<Bf8, false>(bv, 0, saturate, policy) << 8;
    }
    result[lane] = (d[0][lane] & ~selected) | (value << shift);
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Narrow FP16/FP32 to FP8/BF8 for Wave64. Reject invalid controls before changing outputs.
template <bool Bf8, bool Stochastic, bool Half = false>
int convert_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                   const uint32_t *const *a, const uint32_t *const *b,
                   Fp8NarrowPolicy policy = Fp8NarrowPolicy::Gfx12) {
  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error =
          convert<Bf8, Stochastic, Half>(flags, 0, mode, nullptr, nullptr, nullptr, policy)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  for (int half = 0; half < 2; ++half) {
    uint64_t half_instruction_flags = rocjitsu::arith::dpp_half_wave_instruction_flags(mode, half);
    uint32_t *dest = d[0] + half * 32;
    const uint32_t *source_a = a[0] + half * 32, *source_b = b ? b[0] + half * 32 : nullptr;
    if (int error = convert<Bf8, Stochastic, Half>(flags, uint32_t(exec_mask >> (half * 32)),
                                                   half_instruction_flags, &dest, &source_a,
                                                   b ? &source_b : nullptr, policy)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

// CDNA narrows raw FP32 sources. Source/output modifiers are ignored except
// gfx1250 FP8 CLAMP, which selects unsigned E5M3 rather than numeric clamping.
uint64_t cdna_narrow_mode(uint64_t mode) {
  return mode & ~uint64_t(RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_ABS_B |
                          RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_CLAMP | RJ_ARITH_ALU_OMOD_MASK);
}

} // namespace

// Convert F32 to FP8 and pack results with packed elements for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_pk_fp8_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert<false, false>(flags, exec_mask, mode, d, a, b);
}

// Convert F32 to BF8 and pack results with packed elements for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_pk_bf8_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert<true, false>(flags, exec_mask, mode, d, a, b);
}

// Convert F32 to FP8 with seeded stochastic rounding for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_sr_fp8_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert<false, true>(flags, exec_mask, mode, d, a, b);
}

// Convert F32 to BF8 with seeded stochastic rounding for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_sr_bf8_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert<true, true>(flags, exec_mask, mode, d, a, b);
}

// Convert F32 to FP8 and pack results with packed elements for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_pk_fp8_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert_wave64<false, false>(flags, exec_mask, mode, d, a, b);
}

// Convert F32 to BF8 and pack results with packed elements for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_pk_bf8_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert_wave64<true, false>(flags, exec_mask, mode, d, a, b);
}

// Convert F32 to FP8 with seeded stochastic rounding for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_sr_fp8_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert_wave64<false, true>(flags, exec_mask, mode, d, a, b);
}

// Convert F32 to BF8 with seeded stochastic rounding for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_sr_bf8_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert_wave64<true, true>(flags, exec_mask, mode, d, a, b);
}

// Narrow with inherited gfx942 format, overflow and destination-half rules; honor EXEC.
int rj_arith_v_cvt_pk_fp8_f32_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  auto policy = Fp8NarrowPolicy::Fnuz;
  return convert<false, false>(flags, exec_mask, cdna_narrow_mode(mode), d, a, b, policy);
}

// Narrow with inherited gfx942 format, overflow and destination-half rules; honor EXEC.
int rj_arith_v_cvt_pk_fp8_f32_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  auto policy = Fp8NarrowPolicy::Fnuz;
  return convert_wave64<false, false>(flags, exec_mask, cdna_narrow_mode(mode), d, a, b, policy);
}

// Narrow with inherited gfx942 format, overflow and destination-half rules; honor EXEC.
int rj_arith_v_cvt_sr_fp8_f32_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  auto policy = Fp8NarrowPolicy::Fnuz;
  return convert<false, true>(flags, exec_mask, cdna_narrow_mode(mode), d, a, b, policy);
}

// Narrow with inherited gfx942 format, overflow and destination-half rules; honor EXEC.
int rj_arith_v_cvt_sr_fp8_f32_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  auto policy = Fp8NarrowPolicy::Fnuz;
  return convert_wave64<false, true>(flags, exec_mask, cdna_narrow_mode(mode), d, a, b, policy);
}

// Narrow with inherited gfx942 format, overflow and destination-half rules; honor EXEC.
int rj_arith_v_cvt_pk_bf8_f32_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  auto policy = Fp8NarrowPolicy::Fnuz;
  return convert<true, false>(flags, exec_mask, cdna_narrow_mode(mode), d, a, b, policy);
}

// Narrow with inherited gfx942 format, overflow and destination-half rules; honor EXEC.
int rj_arith_v_cvt_pk_bf8_f32_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  auto policy = Fp8NarrowPolicy::Fnuz;
  return convert_wave64<true, false>(flags, exec_mask, cdna_narrow_mode(mode), d, a, b, policy);
}

// Narrow with inherited gfx942 format, overflow and destination-half rules; honor EXEC.
int rj_arith_v_cvt_sr_bf8_f32_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  auto policy = Fp8NarrowPolicy::Fnuz;
  return convert<true, true>(flags, exec_mask, cdna_narrow_mode(mode), d, a, b, policy);
}

// Narrow with inherited gfx942 format, overflow and destination-half rules; honor EXEC.
int rj_arith_v_cvt_sr_bf8_f32_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  auto policy = Fp8NarrowPolicy::Fnuz;
  return convert_wave64<true, true>(flags, exec_mask, cdna_narrow_mode(mode), d, a, b, policy);
}

// Narrow with inherited gfx950 format, overflow and destination-half rules; honor EXEC.
int rj_arith_v_cvt_pk_fp8_f32_w32_gfx950(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  auto policy = Fp8NarrowPolicy::Ocp;
  return convert<false, false>(flags, exec_mask, cdna_narrow_mode(mode), d, a, b, policy);
}

// Narrow with inherited gfx950 format, overflow and destination-half rules; honor EXEC.
int rj_arith_v_cvt_pk_fp8_f32_w64_gfx950(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  auto policy = Fp8NarrowPolicy::Ocp;
  return convert_wave64<false, false>(flags, exec_mask, cdna_narrow_mode(mode), d, a, b, policy);
}

// Narrow with inherited gfx950 format, overflow and destination-half rules; honor EXEC.
int rj_arith_v_cvt_sr_fp8_f32_w32_gfx950(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  auto policy = Fp8NarrowPolicy::Ocp;
  return convert<false, true>(flags, exec_mask, cdna_narrow_mode(mode), d, a, b, policy);
}

// Narrow with inherited gfx950 format, overflow and destination-half rules; honor EXEC.
int rj_arith_v_cvt_sr_fp8_f32_w64_gfx950(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  auto policy = Fp8NarrowPolicy::Ocp;
  return convert_wave64<false, true>(flags, exec_mask, cdna_narrow_mode(mode), d, a, b, policy);
}

// Narrow with inherited gfx950 format, overflow and destination-half rules; honor EXEC.
int rj_arith_v_cvt_pk_bf8_f32_w32_gfx950(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  auto policy = Fp8NarrowPolicy::Ocp;
  return convert<true, false>(flags, exec_mask, cdna_narrow_mode(mode), d, a, b, policy);
}

// Narrow with inherited gfx950 format, overflow and destination-half rules; honor EXEC.
int rj_arith_v_cvt_pk_bf8_f32_w64_gfx950(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  auto policy = Fp8NarrowPolicy::Ocp;
  return convert_wave64<true, false>(flags, exec_mask, cdna_narrow_mode(mode), d, a, b, policy);
}

// Narrow with inherited gfx950 format, overflow and destination-half rules; honor EXEC.
int rj_arith_v_cvt_sr_bf8_f32_w32_gfx950(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  auto policy = Fp8NarrowPolicy::Ocp;
  return convert<true, true>(flags, exec_mask, cdna_narrow_mode(mode), d, a, b, policy);
}

// Narrow with inherited gfx950 format, overflow and destination-half rules; honor EXEC.
int rj_arith_v_cvt_sr_bf8_f32_w64_gfx950(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  auto policy = Fp8NarrowPolicy::Ocp;
  return convert_wave64<true, true>(flags, exec_mask, cdna_narrow_mode(mode), d, a, b, policy);
}

// Narrow with inherited gfx1250 format and overflow rules; preserve unselected bytes and lanes.
int rj_arith_v_cvt_pk_fp8_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                          uint32_t *const *d, const uint32_t *const *a,
                                          const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  auto policy = mode & RJ_ARITH_ALU_CLAMP ? Fp8NarrowPolicy::E5M3 : Fp8NarrowPolicy::OcpPreserve;
  return convert<false, false>(flags, exec_mask, cdna_narrow_mode(mode), d, a, b, policy);
}

// Narrow with inherited gfx1250 format and overflow rules; preserve unselected bytes and lanes.
int rj_arith_v_cvt_pk_fp8_f32_w64_gfx1250(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                          uint32_t *const *d, const uint32_t *const *a,
                                          const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  auto policy = mode & RJ_ARITH_ALU_CLAMP ? Fp8NarrowPolicy::E5M3 : Fp8NarrowPolicy::OcpPreserve;
  return convert_wave64<false, false>(flags, exec_mask, cdna_narrow_mode(mode), d, a, b, policy);
}

// Narrow with inherited gfx1250 format and overflow rules; preserve unselected bytes and lanes.
int rj_arith_v_cvt_sr_fp8_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                          uint32_t *const *d, const uint32_t *const *a,
                                          const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  auto policy = mode & RJ_ARITH_ALU_CLAMP ? Fp8NarrowPolicy::E5M3 : Fp8NarrowPolicy::OcpPreserve;
  return convert<false, true>(flags, exec_mask, cdna_narrow_mode(mode), d, a, b, policy);
}

// Narrow with inherited gfx1250 format and overflow rules; preserve unselected bytes and lanes.
int rj_arith_v_cvt_sr_fp8_f32_w64_gfx1250(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                          uint32_t *const *d, const uint32_t *const *a,
                                          const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  auto policy = mode & RJ_ARITH_ALU_CLAMP ? Fp8NarrowPolicy::E5M3 : Fp8NarrowPolicy::OcpPreserve;
  return convert_wave64<false, true>(flags, exec_mask, cdna_narrow_mode(mode), d, a, b, policy);
}

// Narrow OCP BF8, preserving destination bits outside the selected byte or half.
int rj_arith_v_cvt_pk_bf8_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                          uint32_t *const *d, const uint32_t *const *a,
                                          const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert<true, false>(flags, exec_mask, cdna_narrow_mode(mode), d, a, b,
                              Fp8NarrowPolicy::OcpPreserve);
}

// Narrow OCP BF8, preserving destination bits outside the selected byte or half.
int rj_arith_v_cvt_pk_bf8_f32_w64_gfx1250(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                          uint32_t *const *d, const uint32_t *const *a,
                                          const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert_wave64<true, false>(flags, exec_mask, cdna_narrow_mode(mode), d, a, b,
                                     Fp8NarrowPolicy::OcpPreserve);
}

// Narrow OCP BF8, preserving destination bits outside the selected byte or half.
int rj_arith_v_cvt_sr_bf8_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                          uint32_t *const *d, const uint32_t *const *a,
                                          const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert<true, true>(flags, exec_mask, cdna_narrow_mode(mode), d, a, b,
                             Fp8NarrowPolicy::OcpPreserve);
}

// Narrow OCP BF8, preserving destination bits outside the selected byte or half.
int rj_arith_v_cvt_sr_bf8_f32_w64_gfx1250(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                          uint32_t *const *d, const uint32_t *const *a,
                                          const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert_wave64<true, true>(flags, exec_mask, cdna_narrow_mode(mode), d, a, b,
                                    Fp8NarrowPolicy::OcpPreserve);
}

// Narrow both FP16 halves of A; preserve unselected destination bits and lanes.
int rj_arith_v_cvt_pk_fp8_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  auto policy = mode & RJ_ARITH_ALU_CLAMP ? Fp8NarrowPolicy::E5M3 : Fp8NarrowPolicy::OcpPreserve;
  return convert<false, false, true>(flags, exec_mask, cdna_narrow_mode(mode), d, a, nullptr,
                                     policy);
}

// Narrow both FP16 halves of A; preserve unselected destination bits and lanes.
int rj_arith_v_cvt_pk_fp8_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  auto policy = mode & RJ_ARITH_ALU_CLAMP ? Fp8NarrowPolicy::E5M3 : Fp8NarrowPolicy::OcpPreserve;
  return convert_wave64<false, false, true>(flags, exec_mask, cdna_narrow_mode(mode), d, a, nullptr,
                                            policy);
}

// Narrow the selected FP16 half using seed B; preserve unselected destination bits and lanes.
int rj_arith_v_cvt_sr_fp8_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  auto policy = mode & RJ_ARITH_ALU_CLAMP ? Fp8NarrowPolicy::E5M3 : Fp8NarrowPolicy::OcpPreserve;
  return convert<false, true, true>(flags, exec_mask, cdna_narrow_mode(mode), d, a, b, policy);
}

// Narrow the selected FP16 half using seed B; preserve unselected destination bits and lanes.
int rj_arith_v_cvt_sr_fp8_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  auto policy = mode & RJ_ARITH_ALU_CLAMP ? Fp8NarrowPolicy::E5M3 : Fp8NarrowPolicy::OcpPreserve;
  return convert_wave64<false, true, true>(flags, exec_mask, cdna_narrow_mode(mode), d, a, b,
                                           policy);
}

// Narrow both FP16 halves of A; preserve unselected destination bits and lanes.
int rj_arith_v_cvt_pk_bf8_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  auto policy = Fp8NarrowPolicy::OcpPreserve;
  return convert<true, false, true>(flags, exec_mask, cdna_narrow_mode(mode), d, a, nullptr,
                                    policy);
}

// Narrow both FP16 halves of A; preserve unselected destination bits and lanes.
int rj_arith_v_cvt_pk_bf8_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  auto policy = Fp8NarrowPolicy::OcpPreserve;
  return convert_wave64<true, false, true>(flags, exec_mask, cdna_narrow_mode(mode), d, a, nullptr,
                                           policy);
}

// Narrow the selected FP16 half using seed B; preserve unselected destination bits and lanes.
int rj_arith_v_cvt_sr_bf8_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  auto policy = Fp8NarrowPolicy::OcpPreserve;
  return convert<true, true, true>(flags, exec_mask, cdna_narrow_mode(mode), d, a, b, policy);
}

// Narrow the selected FP16 half using seed B; preserve unselected destination bits and lanes.
int rj_arith_v_cvt_sr_bf8_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  auto policy = Fp8NarrowPolicy::OcpPreserve;
  return convert_wave64<true, true, true>(flags, exec_mask, cdna_narrow_mode(mode), d, a, b,
                                          policy);
}
