// RUN: %clang_cc1 %s -ffake-address-space-map -faddress-space-map-mangling=no -triple %itanium_abi_triple -emit-llvm -o - | FileCheck %s

__kernel void foo(void) {
  // CHECK: @foo.i = internal addrspace(2)
  __local int i;
  ++i;
}

// CHECK-LABEL: define void @_Z3barPU7CLlocali
__kernel void __attribute__((__overloadable__)) bar(local int *x) {
  *x = 5;
}
