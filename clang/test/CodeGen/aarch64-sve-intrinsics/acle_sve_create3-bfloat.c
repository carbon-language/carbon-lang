// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - -x c++ %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - -x c++ %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify -verify-ignore-unexpected=error -verify-ignore-unexpected=note %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

svbfloat16x3_t test_svcreate3_bf16(svbfloat16_t x0, svbfloat16_t x1, svbfloat16_t x2)
{
  // CHECK-LABEL: test_svcreate3_bf16
  // CHECK: %[[CREATE:.*]] = call <vscale x 24 x bfloat> @llvm.aarch64.sve.tuple.create3.nxv24bf16.nxv8bf16(<vscale x 8 x bfloat> %x0, <vscale x 8 x bfloat> %x1, <vscale x 8 x bfloat> %x2)
  // CHECK-NEXT: ret <vscale x 24 x bfloat> %[[CREATE]]
  // expected-warning@+1 {{implicit declaration of function 'svcreate3_bf16'}}
  return SVE_ACLE_FUNC(svcreate3,_bf16,,)(x0, x1, x2);
}
