// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify -verify-ignore-unexpected=note %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify -verify-ignore-unexpected=note %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

svint8_t test_svget4_s8(svint8x4_t tuple)
{
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
  return SVE_ACLE_FUNC(svget4,_s8,,)(tuple, -1);
}

svint16_t test_svget4_s16(svint16x4_t tuple)
{
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return SVE_ACLE_FUNC(svget4,_s16,,)(tuple, 4);
}

svint32_t test_svget4_s32(svint32x4_t tuple)
{
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
  return SVE_ACLE_FUNC(svget4,_s32,,)(tuple, -1);
}

svint64_t test_svget4_s64(svint64x4_t tuple)
{
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return SVE_ACLE_FUNC(svget4,_s64,,)(tuple, 4);
}

svuint8_t test_svget4_u8(svuint8x4_t tuple)
{
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
  return SVE_ACLE_FUNC(svget4,_u8,,)(tuple, -1);
}

svuint16_t test_svget4_u16(svuint16x4_t tuple)
{
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return SVE_ACLE_FUNC(svget4,_u16,,)(tuple, 4);
}

svuint32_t test_svget4_u32(svuint32x4_t tuple)
{
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
  return SVE_ACLE_FUNC(svget4,_u32,,)(tuple, -1);
}

svuint64_t test_svget4_u64(svuint64x4_t tuple)
{
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return SVE_ACLE_FUNC(svget4,_u64,,)(tuple, 4);
}

svfloat16_t test_svget4_f16(svfloat16x4_t tuple)
{
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
  return SVE_ACLE_FUNC(svget4,_f16,,)(tuple, -1);
}

svfloat32_t test_svget4_f32(svfloat32x4_t tuple)
{
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return SVE_ACLE_FUNC(svget4,_f32,,)(tuple, 4);
}

svfloat64_t test_svget4_f64(svfloat64x4_t tuple)
{
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
  return SVE_ACLE_FUNC(svget4,_f64,,)(tuple, -1);
}

svint8_t test_svget4_s8_var(svint8x4_t tuple, uint64_t imm_index)
{
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  return SVE_ACLE_FUNC(svget4,_s8,,)(tuple, imm_index);
}

svint16_t test_svget4_s16_var(svint16x4_t tuple, uint64_t imm_index)
{
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  return SVE_ACLE_FUNC(svget4,_s16,,)(tuple, imm_index);
}

svint32_t test_svget4_s32_var(svint32x4_t tuple, uint64_t imm_index)
{
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  return SVE_ACLE_FUNC(svget4,_s32,,)(tuple, imm_index);
}

svint64_t test_svget4_s64_var(svint64x4_t tuple, uint64_t imm_index)
{
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  return SVE_ACLE_FUNC(svget4,_s64,,)(tuple, imm_index);
}

svuint8_t test_svget4_u8_var(svuint8x4_t tuple, uint64_t imm_index)
{
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  return SVE_ACLE_FUNC(svget4,_u8,,)(tuple, imm_index);
}

svuint16_t test_svget4_u16_var(svuint16x4_t tuple, uint64_t imm_index)
{
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  return SVE_ACLE_FUNC(svget4,_u16,,)(tuple, imm_index);
}

svuint32_t test_svget4_u32_var(svuint32x4_t tuple, uint64_t imm_index)
{
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  return SVE_ACLE_FUNC(svget4,_u32,,)(tuple, imm_index);
}

svuint64_t test_svget4_u64_var(svuint64x4_t tuple, uint64_t imm_index)
{
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  return SVE_ACLE_FUNC(svget4,_u64,,)(tuple, imm_index);
}

svfloat16_t test_svget4_f16_var(svfloat16x4_t tuple, uint64_t imm_index)
{
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  return SVE_ACLE_FUNC(svget4,_f16,,)(tuple, imm_index);
}

svfloat32_t test_svget4_f32_var(svfloat32x4_t tuple, uint64_t imm_index)
{
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  return SVE_ACLE_FUNC(svget4,_f32,,)(tuple, imm_index);
}

svfloat64_t test_svget4_f64_var(svfloat64x4_t tuple, uint64_t imm_index)
{
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  return SVE_ACLE_FUNC(svget4,_f64,,)(tuple, imm_index);
}
