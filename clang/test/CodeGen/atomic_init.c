// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s

// Check that no atomic operations are used in any initialisation of _Atomic
// types.  

_Atomic(int) i = 42;

void foo()
{
  _Atomic(int) j = 12; // CHECK: store 
                       // CHECK-NOT: atomic
  __c11_atomic_init(&j, 42); // CHECK: store 
                             // CHECK-NOT: atomic
}
