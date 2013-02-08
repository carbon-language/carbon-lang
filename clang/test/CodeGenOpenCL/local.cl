// RUN: %clang_cc1 %s -ffake-address-space-map -emit-llvm -o - | FileCheck %s

__kernel void foo(void) {
  // CHECK: @foo.i = internal addrspace(2)
  __local int i;
  ++i;
}

// CHECK: define void @_Z3barPU3AS2i
__kernel void __attribute__((__overloadable__)) bar(local int *x) {
  *x = 5;
}
