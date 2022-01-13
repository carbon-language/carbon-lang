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

svint16_t test_svmovlb_s16(svint8_t op1)
{
  // CHECK-LABEL: test_svmovlb_s16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.sshllb.nxv8i16(<vscale x 16 x i8> %op1, i32 0)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svmovlb'}}
  // expected-warning@+1 {{implicit declaration of function 'svmovlb_s16'}}
  return SVE_ACLE_FUNC(svmovlb,_s16,,)(op1);
}

svint32_t test_svmovlb_s32(svint16_t op1)
{
  // CHECK-LABEL: test_svmovlb_s32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sshllb.nxv4i32(<vscale x 8 x i16> %op1, i32 0)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svmovlb'}}
  // expected-warning@+1 {{implicit declaration of function 'svmovlb_s32'}}
  return SVE_ACLE_FUNC(svmovlb,_s32,,)(op1);
}

svint64_t test_svmovlb_s64(svint32_t op1)
{
  // CHECK-LABEL: test_svmovlb_s64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.sshllb.nxv2i64(<vscale x 4 x i32> %op1, i32 0)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svmovlb'}}
  // expected-warning@+1 {{implicit declaration of function 'svmovlb_s64'}}
  return SVE_ACLE_FUNC(svmovlb,_s64,,)(op1);
}

svuint16_t test_svmovlb_u16(svuint8_t op1)
{
  // CHECK-LABEL: test_svmovlb_u16
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.ushllb.nxv8i16(<vscale x 16 x i8> %op1, i32 0)
  // CHECK: ret <vscale x 8 x i16> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svmovlb'}}
  // expected-warning@+1 {{implicit declaration of function 'svmovlb_u16'}}
  return SVE_ACLE_FUNC(svmovlb,_u16,,)(op1);
}

svuint32_t test_svmovlb_u32(svuint16_t op1)
{
  // CHECK-LABEL: test_svmovlb_u32
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.ushllb.nxv4i32(<vscale x 8 x i16> %op1, i32 0)
  // CHECK: ret <vscale x 4 x i32> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svmovlb'}}
  // expected-warning@+1 {{implicit declaration of function 'svmovlb_u32'}}
  return SVE_ACLE_FUNC(svmovlb,_u32,,)(op1);
}

svuint64_t test_svmovlb_u64(svuint32_t op1)
{
  // CHECK-LABEL: test_svmovlb_u64
  // CHECK: %[[INTRINSIC:.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.ushllb.nxv2i64(<vscale x 4 x i32> %op1, i32 0)
  // CHECK: ret <vscale x 2 x i64> %[[INTRINSIC]]
  // overload-warning@+2 {{implicit declaration of function 'svmovlb'}}
  // expected-warning@+1 {{implicit declaration of function 'svmovlb_u64'}}
  return SVE_ACLE_FUNC(svmovlb,_u64,,)(op1);
}
