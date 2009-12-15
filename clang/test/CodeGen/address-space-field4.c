// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s
// CHECK: addrspace(2)
// CHECK: addrspace(3)
// CHECK: addrspace(1)
// CHECK: addrspace(1)
// CHECK: addrspace(1)
// CHECK: addrspace(1)
// CHECK: addrspace(1)
// CHECK: addrspace(1)
// CHECK: addrspace(1)
// CHECK: addrspace(1)
// CHECK: addrspace(1)
// CHECK: addrspace(1)
// CHECK: addrspace(1)
// CHECK: addrspace(1)
// CHECK: addrspace(1)
// CHECK: addrspace(1)
// CHECK: addrspace(1)
// CHECK: addrspace(1)
// CHECK: addrspace(1)
// CHECK: addrspace(1)
// CHECK: addrspace(1)
// CHECK: addrspace(3)
// CHECK: addrspace(3)
// CHECK: addrspace(1)
// CHECK: addrspace(3)
// CHECK: addrspace(3)
// CHECK: addrspace(1)
// CHECK: addrspace(1)
// CHECK: addrspace(1)
// CHECK: addrspace(1)
// CHECK: addrspace(1)
// CHECK: addrspace(1)
// CHECK: addrspace(1)
// CHECK: addrspace(1)
// CHECK: addrspace(1)
// CHECK: addrspace(2)
// CHECK: addrspace(1)
// CHECK: addrspace(2)
// CHECK: addrspace(2)
// CHECK: addrspace(2)

// Check the load and store are using the correct address space to access
// the variables.

#define __addr1    __attribute__((address_space(1)))
#define __addr2    __attribute__((address_space(2)))
#define __addr3    __attribute__((address_space(3)))

typedef struct Pair {
  __addr2 int* a;
  __addr3 int* b;
} Pair;

typedef struct S {
  Pair arr[ 3 ];
} S;

void test_addrspace(__addr1 S* p1, __addr1 S* p2) {
  *p1->arr[0].a = *p2->arr[1].b;
}
