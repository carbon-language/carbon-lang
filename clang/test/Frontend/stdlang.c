// RUN: %clang_cc1 -x cuda -std=c++11 -DCUDA %s
// RUN: %clang_cc1 -x cl -std=c99 -DOPENCL %s
// expected-no-diagnostics

#if defined(CUDA)
  __attribute__((device)) void f_device();
#elif defined(OPENCL)
  kernel void func(void);
#endif
