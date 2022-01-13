// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve2 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve2 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - -x c++ %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve2 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve2 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - -x c++ %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify -verify-ignore-unexpected=error %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify=overload -verify-ignore-unexpected=error %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

svbool_t test_svwhilerw_s8(const int8_t *op1, const int8_t *op2)
{
  // CHECK-LABEL: test_svwhilerw_s8
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.whilerw.b.nxv16i1.p0i8(i8* %op1, i8* %op2)
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svwhilerw'}}
  // expected-warning@+1 {{implicit declaration of function 'svwhilerw_s8'}}
  return SVE_ACLE_FUNC(svwhilerw,_s8,,)(op1, op2);
}

svbool_t test_svwhilerw_s16(const int16_t *op1, const int16_t *op2)
{
  // CHECK-LABEL: test_svwhilerw_s16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.whilerw.h.nxv8i1.p0i16(i16* %op1, i16* %op2)
  // CHECK: %[[INTRINSIC_REINT:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC_REINT]]
  // overload-warning@+2 {{implicit declaration of function 'svwhilerw'}}
  // expected-warning@+1 {{implicit declaration of function 'svwhilerw_s16'}}
  return SVE_ACLE_FUNC(svwhilerw,_s16,,)(op1, op2);
}

svbool_t test_svwhilerw_s32(const int32_t *op1, const int32_t *op2)
{
  // CHECK-LABEL: test_svwhilerw_s32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.whilerw.s.nxv4i1.p0i32(i32* %op1, i32* %op2)
  // CHECK: %[[INTRINSIC_REINT:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC_REINT]]
  // overload-warning@+2 {{implicit declaration of function 'svwhilerw'}}
  // expected-warning@+1 {{implicit declaration of function 'svwhilerw_s32'}}
  return SVE_ACLE_FUNC(svwhilerw,_s32,,)(op1, op2);
}

svbool_t test_svwhilerw_s64(const int64_t *op1, const int64_t *op2)
{
  // CHECK-LABEL: test_svwhilerw_s64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.whilerw.d.nxv2i1.p0i64(i64* %op1, i64* %op2)
  // CHECK: %[[INTRINSIC_REINT:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC_REINT]]
  // overload-warning@+2 {{implicit declaration of function 'svwhilerw'}}
  // expected-warning@+1 {{implicit declaration of function 'svwhilerw_s64'}}
  return SVE_ACLE_FUNC(svwhilerw,_s64,,)(op1, op2);
}

svbool_t test_svwhilerw_u8(const uint8_t *op1, const uint8_t *op2)
{
  // CHECK-LABEL: test_svwhilerw_u8
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.whilerw.b.nxv16i1.p0i8(i8* %op1, i8* %op2)
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svwhilerw'}}
  // expected-warning@+1 {{implicit declaration of function 'svwhilerw_u8'}}
  return SVE_ACLE_FUNC(svwhilerw,_u8,,)(op1, op2);
}

svbool_t test_svwhilerw_u16(const uint16_t *op1, const uint16_t *op2)
{
  // CHECK-LABEL: test_svwhilerw_u16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.whilerw.h.nxv8i1.p0i16(i16* %op1, i16* %op2)
  // CHECK: %[[INTRINSIC_REINT:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC_REINT]]
  // overload-warning@+2 {{implicit declaration of function 'svwhilerw'}}
  // expected-warning@+1 {{implicit declaration of function 'svwhilerw_u16'}}
  return SVE_ACLE_FUNC(svwhilerw,_u16,,)(op1, op2);
}

svbool_t test_svwhilerw_u32(const uint32_t *op1, const uint32_t *op2)
{
  // CHECK-LABEL: test_svwhilerw_u32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.whilerw.s.nxv4i1.p0i32(i32* %op1, i32* %op2)
  // CHECK: %[[INTRINSIC_REINT:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC_REINT]]
  // overload-warning@+2 {{implicit declaration of function 'svwhilerw'}}
  // expected-warning@+1 {{implicit declaration of function 'svwhilerw_u32'}}
  return SVE_ACLE_FUNC(svwhilerw,_u32,,)(op1, op2);
}

svbool_t test_svwhilerw_u64(const uint64_t *op1, const uint64_t *op2)
{
  // CHECK-LABEL: test_svwhilerw_u64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.whilerw.d.nxv2i1.p0i64(i64* %op1, i64* %op2)
  // CHECK: %[[INTRINSIC_REINT:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC_REINT]]
  // overload-warning@+2 {{implicit declaration of function 'svwhilerw'}}
  // expected-warning@+1 {{implicit declaration of function 'svwhilerw_u64'}}
  return SVE_ACLE_FUNC(svwhilerw,_u64,,)(op1, op2);
}

svbool_t test_svwhilerw_f16(const float16_t *op1, const float16_t *op2)
{
  // CHECK-LABEL: test_svwhilerw_f16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.whilerw.h.nxv8i1.p0f16(half* %op1, half* %op2)
  // CHECK: %[[INTRINSIC_REINT:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC_REINT]]
  // overload-warning@+2 {{implicit declaration of function 'svwhilerw'}}
  // expected-warning@+1 {{implicit declaration of function 'svwhilerw_f16'}}
  return SVE_ACLE_FUNC(svwhilerw,_f16,,)(op1, op2);
}

svbool_t test_svwhilerw_f32(const float32_t *op1, const float32_t *op2)
{
  // CHECK-LABEL: test_svwhilerw_f32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i1> @llvm.aarch64.sve.whilerw.s.nxv4i1.p0f32(float* %op1, float* %op2)
  // CHECK: %[[INTRINSIC_REINT:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %[[INTRINSIC]])
  // CHECK: ret <vscale x 16 x i1> %[[INTRINSIC_REINT]]
  // overload-warning@+2 {{implicit declaration of function 'svwhilerw'}}
  // expected-warning@+1 {{implicit declaration of function 'svwhilerw_f32'}}
  return SVE_ACLE_FUNC(svwhilerw,_f32,,)(op1, op2);
}

svbool_t test_svwhilerw_f64(const float64_t *op1, const float64_t *op2)
{
  // CHECK-LABEL: test_svwhilerw_f64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i1> @llvm.aarch64.sve.whilerw.d.nxv2i1.p0f64(double* %op1, double* %op2)
  // CHECK: %[[INTRINSIC_REINT:.*]] = call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %[[INTRINSIC]])
  // overload-warning@+2 {{implicit declaration of function 'svwhilerw'}}
  // expected-warning@+1 {{implicit declaration of function 'svwhilerw_f64'}}
  return SVE_ACLE_FUNC(svwhilerw,_f64,,)(op1, op2);
}
