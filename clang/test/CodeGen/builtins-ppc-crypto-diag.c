// REQUIRES: powerpc-registered-target
// RUN: not %clang_cc1 -faltivec -triple powerpc64le-unknown-unknown -D_T1LW -target-feature +crypto -emit-llvm %s -o - 2>&1 | FileCheck %s -check-prefix=CHECK-T1
// RUN: not %clang_cc1 -faltivec -triple powerpc64le-unknown-unknown -D_T1MW -target-feature +crypto -emit-llvm %s -o - 2>&1 | FileCheck %s -check-prefix=CHECK-T1
// RUN: not %clang_cc1 -faltivec -triple powerpc64le-unknown-unknown -D_T2LW -target-feature +crypto -emit-llvm %s -o - 2>&1 | FileCheck %s -check-prefix=CHECK-T2
// RUN: not %clang_cc1 -faltivec -triple powerpc64le-unknown-unknown -D_T2MW -target-feature +crypto -emit-llvm %s -o - 2>&1 | FileCheck %s -check-prefix=CHECK-T2
// RUN: not %clang_cc1 -faltivec -triple powerpc64le-unknown-unknown -D_T1LD -target-feature +crypto -emit-llvm %s -o - 2>&1 | FileCheck %s -check-prefix=CHECK-T1
// RUN: not %clang_cc1 -faltivec -triple powerpc64le-unknown-unknown -D_T1MD -target-feature +crypto -emit-llvm %s -o - 2>&1 | FileCheck %s -check-prefix=CHECK-T1
// RUN: not %clang_cc1 -faltivec -triple powerpc64le-unknown-unknown -D_T2LD -target-feature +crypto -emit-llvm %s -o - 2>&1 | FileCheck %s -check-prefix=CHECK-T2
// RUN: not %clang_cc1 -faltivec -triple powerpc64le-unknown-unknown -D_T2MD -target-feature +crypto -emit-llvm %s -o - 2>&1 | FileCheck %s -check-prefix=CHECK-T2
#include <altivec.h>

#define W_INIT { 0x01020304, 0x05060708, \
                  0x090A0B0C, 0x0D0E0F10 };
#define D_INIT { 0x0102030405060708, \
                  0x090A0B0C0D0E0F10 };
vector unsigned int test_vshasigmaw_or(void)
{
  vector unsigned int a = W_INIT
#ifdef _T1LW // Arg1 too large
  vector unsigned int b = __builtin_crypto_vshasigmaw(a, 2, 15);
#elif defined(_T1MW) // Arg1 negative
  vector unsigned int c = __builtin_crypto_vshasigmaw(a, -1, 15);
#elif defined(_T2LW) // Arg2 too large
  vector unsigned int d = __builtin_crypto_vshasigmaw(a, 0, 85);
#elif defined(_T2MW) // Arg1 negative
  vector unsigned int e = __builtin_crypto_vshasigmaw(a, 1, -15);
#endif
  return __builtin_crypto_vshasigmaw(a, 1, 15);
}

vector unsigned long long test_vshasigmad_or(void)
{
  vector unsigned long long a = D_INIT
#ifdef _T1LD // Arg1 too large
  vector unsigned long long b = __builtin_crypto_vshasigmad(a, 2, 15);
#elif defined(_T1MD) // Arg1 negative
  vector unsigned long long c = __builtin_crypto_vshasigmad(a, -1, 15);
#elif defined(_T2LD) // Arg2 too large
  vector unsigned long long d = __builtin_crypto_vshasigmad(a, 0, 85);
#elif defined(_T2MD) // Arg1 negative
  vector unsigned long long e = __builtin_crypto_vshasigmad(a, 1, -15);
#endif
  return __builtin_crypto_vshasigmad(a, 0, 15);
}

// CHECK-T1: error: argument out of range (should be 0-1).
// CHECK-T2: error: argument out of range (should be 0-15).
