// Tests that target-specific builtins have appropriate host/device
// attributes and that CUDA call restrictions are enforced. Also
// verify that non-target builtins can be used from both host and
// device functions.
//
// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target
// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN:     -fcuda-target-overloads -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple nvptx64-unknown-cuda -fcuda-is-device \
// RUN:     -fcuda-target-overloads -fsyntax-only -verify %s


#ifdef __CUDA_ARCH__
// Device-side builtins are not allowed to be called from host functions.
void hf() {
  int x = __builtin_ptx_read_tid_x(); // expected-note  {{'__builtin_ptx_read_tid_x' declared here}}
  // expected-error@-1 {{reference to __device__ function '__builtin_ptx_read_tid_x' in __host__ function}}
  x = __builtin_abs(1);
}
__attribute__((device)) void df() {
  int x = __builtin_ptx_read_tid_x();
  x = __builtin_abs(1);
}
#else
// Host-side builtins are not allowed to be called from device functions.
__attribute__((device)) void df() {
  int x = __builtin_ia32_rdtsc();   // expected-note {{'__builtin_ia32_rdtsc' declared here}}
  // expected-error@-1 {{reference to __host__ function '__builtin_ia32_rdtsc' in __device__ function}}
  x = __builtin_abs(1);
}
void hf() {
  int x = __builtin_ia32_rdtsc();
  x = __builtin_abs(1);
}
#endif
