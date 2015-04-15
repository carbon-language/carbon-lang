// Test that we can disable cross-target call checks in Sema with the
// -fcuda-disable-target-call-checks flag. Without this flag we'd get a bunch
// of errors here, since there are invalid cross-target calls present.

// RUN: %clang_cc1 -fsyntax-only -verify %s -fcuda-disable-target-call-checks
// RUN: %clang_cc1 -fsyntax-only -fcuda-is-device -verify %s -fcuda-disable-target-call-checks

// expected-no-diagnostics

#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __host__ __attribute__((host))

__attribute__((host)) void h1();

__attribute__((device)) void d1() {
  h1();
}

__attribute__((host)) void h2() {
  d1();
}

__attribute__((global)) void g1() {
  h2();
}
