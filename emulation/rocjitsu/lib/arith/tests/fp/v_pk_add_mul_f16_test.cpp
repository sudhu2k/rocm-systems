// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for packed FP16 addition and multiplication.

#include "common/cpu.h"
#include "fp/binary_arithmetic_test.h"
#include "rj_arith/rj_arith.h"
#include <algorithm>
#include <random>

#include <gtest/gtest.h>

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpPkAddF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32, /*Packed=*/true>(
      rj_arith_v_pk_add_f16_w32_gfx12, [](double x, double y) { return x + y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpPkAddF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32, /*Packed=*/true>(
      rj_arith_v_pk_add_f16_w32, [](double x, double y) { return x + y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpPkAddF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64, /*Packed=*/true>(
      rj_arith_v_pk_add_f16_w64_gfx12, [](double x, double y) { return x + y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpPkAddF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64, /*Packed=*/true>(
      rj_arith_v_pk_add_f16_w64, [](double x, double y) { return x + y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpPkMulF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32, /*Packed=*/true>(
      rj_arith_v_pk_mul_f16_w32_gfx12, [](double x, double y) { return x * y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpPkMulF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32, /*Packed=*/true>(
      rj_arith_v_pk_mul_f16_w32, [](double x, double y) { return x * y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpPkMulF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64, /*Packed=*/true>(
      rj_arith_v_pk_mul_f16_w64_gfx12, [](double x, double y) { return x * y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpPkMulF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64, /*Packed=*/true>(
      rj_arith_v_pk_mul_f16_w64, [](double x, double y) { return x * y; }, /*gfx12=*/false);
}

// The packed guest path applies flushing and finite-overflow saturation to both halves.
TEST(FpPackedBinaryF16, GuestControls) {
  using namespace rocjitsu::arith::test;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (auto call : {rj_arith_v_pk_mul_f16_w32, rj_arith_v_pk_mul_f16_w32_gfx12}) {
      Registers<3, 32> r;
      r.v[0][0] = 0x00010001U;
      r.v[1][0] = 0x3c003c00U;
      ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_LOOSE | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS, 1, 0,
                     r.v + 2, r.v, r.v + 1, nullptr),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(r.v[2][0], 0U);
      r.v[0][0] = 0x7bff7bffU;
      r.v[1][0] = 0x40004000U;
      ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_LOOSE | RJ_ARITH_FP16_OVFL, 1, 0, r.v + 2, r.v,
                     r.v + 1, nullptr),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(r.v[2][0], 0x7bff7bffU);
    }
  }
}

// gfx11 packed arithmetic retains input sNaNs unless IEEE requests quieting;
// ordinary FP16 arithmetic always quiets them in Compatibility.
TEST(FpPackedBinaryF16, CompatibilityNaNs) {
  using namespace rocjitsu::arith::test;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t rounding : {uint64_t{0}, RJ_ARITH_FP_ROUND_ZERO}) {
      for (uint64_t ieee : {uint64_t{0}, RJ_ARITH_FP_IEEE_MODE}) {
        for (auto call : {rj_arith_v_pk_add_f16_w32, rj_arith_v_pk_mul_f16_w32}) {
          Registers<3, 32> r;
          r.v[0][0] = 0x3c003c00U;
          r.v[1][0] = 0x7d017c01U;
          for (uint32_t instruction_flags : {0U, RJ_ARITH_PK_CLAMP}) {
            ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY | rounding | ieee |
                               RJ_ARITH_FP_DX10_CLAMP_OFF,
                           1, instruction_flags, r.v + 2, r.v, r.v + 1, nullptr),
                      RJ_ARITH_SUCCESS);
            EXPECT_EQ(r.v[2][0], ieee ? 0x7f017e01U : 0x7d017c01U);
          }
        }
        for (auto call : {rj_arith_v_add_f16_w32, rj_arith_v_mul_f16_w32}) {
          Registers<3, 32> r;
          r.v[0][0] = 0x3c00U;
          r.v[1][0] = 0x7c01U;
          ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY | rounding | ieee, 1, 0, r.v + 2,
                         r.v, r.v + 1, nullptr),
                    RJ_ARITH_SUCCESS);
          EXPECT_EQ(r.v[2][0] & 0xffffU, 0x7e01U);
        }
      }
    }
  }
}

// The SIMD backends retain the inherited scalar Compatibility model across guest modes.
TEST(HalfArithmetic, SimdGuestModes) {
  struct Case {
    decltype(&rj_arith_v_add_f16_w32) call;
    bool packed, legacy;
  };
  const Case cases[] = {{rj_arith_v_add_f16_w32, false, true},
                        {rj_arith_v_sub_f16_w32, false, true},
                        {rj_arith_v_subrev_f16_w32, false, true},
                        {rj_arith_v_mul_f16_w32, false, true},
                        {rj_arith_v_add_f16_w32_gfx12, false, false},
                        {rj_arith_v_sub_f16_w32_gfx12, false, false},
                        {rj_arith_v_subrev_f16_w32_gfx12, false, false},
                        {rj_arith_v_mul_f16_w32_gfx12, false, false},
                        {rj_arith_v_pk_add_f16_w32, true, true},
                        {rj_arith_v_pk_mul_f16_w32, true, true},
                        {rj_arith_v_pk_add_f16_w32_gfx12, true, false},
                        {rj_arith_v_pk_mul_f16_w32_gfx12, true, false}};
  const auto levels = rocjitsu::arith::test::supported_cpu_levels();
  std::minstd_rand random;
  const uint32_t special[] = {0x00008000, 0x80000000, 0x000103ff,
                              0x7bff7bff, 0x7c00fc00, 0x7d237e12};
  for (const auto &test : cases) {
    for (unsigned rounding = 0; rounding < 4; ++rounding) {
      for (unsigned control = 0; control < 16; ++control) {
        const uint64_t flags = RJ_ARITH_SEMANTICS_COMPATIBILITY | (uint64_t(rounding) << 24) |
                               (control & 1 ? RJ_ARITH_FP_FLUSH_INPUT_DENORMALS : 0) |
                               (control & 2 ? RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS : 0) |
                               (control & 4 ? RJ_ARITH_FP16_OVFL : 0) |
                               (control & 8 && test.legacy ? RJ_ARITH_FP_DX10_CLAMP_OFF : 0);
        for (unsigned variant = 0; variant < 8; ++variant) {
          const uint32_t modifiers =
              test.packed ? (variant & 1 ? RJ_ARITH_PK_NEG_LO_A | RJ_ARITH_PK_NEG_HI_B : 0) |
                                (variant & 2 ? RJ_ARITH_PK_CLAMP : 0) |
                                (variant & 4 ? RJ_ARITH_PK_LO_A_HIGH | RJ_ARITH_PK_HI_B_LOW : 0)
                          : (variant % 4) * RJ_ARITH_ALU_OMOD_2 |
                                (variant & 4 ? RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B |
                                                   RJ_ARITH_ALU_HIGH_D | RJ_ARITH_ALU_NEG_A |
                                                   RJ_ARITH_ALU_CLAMP
                                             : 0);
          for (int batch = 0; batch < 4; ++batch) {
            uint32_t a[32], b[32], reference[32], actual[32];
            for (int lane = 0; lane < 32; ++lane) {
              a[lane] = lane < 6 ? special[lane]
                                 : (uint32_t(random() >> 15) << 16) | uint16_t(random() >> 15);
              b[lane] = lane < 6 ? special[(lane + batch) % 6]
                                 : (uint32_t(random() >> 15) << 16) | uint16_t(random() >> 15);
              reference[lane] = a[lane];
            }
            const uint32_t *x = a, *y = b;
            uint32_t *d = reference, status = 0x80;
            const uint32_t exec_mask = batch & 1 ? 0xa5555555U : 0xffffffffU;
            ASSERT_EQ(test.call(flags, exec_mask, modifiers, &d, &x, &y, &status),
                      RJ_ARITH_SUCCESS);
            for (uint64_t cpu : levels) {
              std::copy(a, a + 32, actual);
              d = actual;
              x = actual;
              ASSERT_EQ(test.call(flags | cpu, exec_mask, modifiers, &d, &x, &y, &status),
                        RJ_ARITH_SUCCESS);
              for (int lane = 0; lane < 32; ++lane) {
                ASSERT_EQ(actual[lane], reference[lane])
                    << int(&test - cases) << '/' << cpu << '/' << rounding << '/' << control << '/'
                    << variant << '/' << batch << '/' << lane;
              }
              EXPECT_EQ(status, 0x80U);
            }
          }
        }
      }
    }
  }
}
