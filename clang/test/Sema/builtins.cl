// RUN: %clang_cc1 %s -fsyntax-only -verify -pedantic
// expected-no-diagnostics

kernel void test(global float *out, global float *in, global int* in2) {
  out[0] = __builtin_nanf("");
  __builtin_memcpy(out, in, 32);
  out[0] = __builtin_frexpf(in[0], in2);
}
