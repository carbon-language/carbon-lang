// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

struct S {
  double a;
  long double b;
};

// CHECK: %struct.{{[a-zA-Z0-9]+}} = type { double, ppc_fp128 }

long double test (struct S x)
{
  return x.b;
}

// CHECK: %{{[0-9]}} = load ppc_fp128, ppc_fp128* %{{[a-zA-Z0-9]+}}, align 16
