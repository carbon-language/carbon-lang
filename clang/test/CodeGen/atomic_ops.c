// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s

void foo(void)
{
  _Atomic(int) i = 0;
  // Check that multiply / divides on atomics produce a cmpxchg loop
  i *= 2; // CHECK: cmpxchg
  i /= 2; // CHECK: cmpxchg
  // These should be emitting atomicrmw instructions, but they aren't yet
  i += 2; // CHECK: cmpxchg
  i -= 2; // CHECK: cmpxchg
  i++; // CHECK: cmpxchg
  i--; // CHECK: cmpxchg
}
