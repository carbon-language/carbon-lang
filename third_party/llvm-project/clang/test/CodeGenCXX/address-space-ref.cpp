// RUN: %clang_cc1 %s -std=c++11 -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s -check-prefixes=CHECK,NULL-INVALID
// RUN: %clang_cc1 %s -std=c++11 -triple=x86_64-apple-darwin10 -fno-delete-null-pointer-checks -emit-llvm -o - | FileCheck %s -check-prefixes=CHECK,NULL-VALID

// For a reference to a complete type, output the dereferenceable attribute (in
// any address space).

typedef int a __attribute__((address_space(1)));

a & foo(a &x, a & y) {
  return x;
}

// CHECK: define{{.*}} align 4 dereferenceable(4) i32 addrspace(1)* @_Z3fooRU3AS1iS0_(i32 addrspace(1)* noundef align 4 dereferenceable(4) %x, i32 addrspace(1)* noundef align 4 dereferenceable(4) %y)

// For a reference to an incomplete type in an alternate address space, output
// neither dereferenceable nor nonnull.

class bc;
typedef bc b __attribute__((address_space(1)));

b & bar(b &x, b & y) {
  return x;
}

// CHECK: define{{.*}} align 1 %class.bc addrspace(1)* @_Z3barRU3AS12bcS1_(%class.bc addrspace(1)* noundef align 1 %x, %class.bc addrspace(1)* noundef align 1 %y)

// For a reference to an incomplete type in addrspace(0), output nonnull.

bc & bar2(bc &x, bc & y) {
  return x;
}

// NULL-INVALID: define{{.*}} nonnull align 1 %class.bc* @_Z4bar2R2bcS0_(%class.bc* noundef nonnull align 1 %x, %class.bc* noundef nonnull align 1 %y)
// NULL-VALID: define{{.*}} align 1 %class.bc* @_Z4bar2R2bcS0_(%class.bc* noundef align 1 %x, %class.bc* noundef align 1 %y)


