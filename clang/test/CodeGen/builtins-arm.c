// RUN: %clang_cc1 -triple thumbv7-eabi -target-cpu cortex-a8 -O3 -emit-llvm -o %t %s

void *f0()
{
  return __builtin_thread_pointer();
}
