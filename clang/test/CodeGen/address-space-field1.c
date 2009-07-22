// RUN: clang-cc -emit-llvm < %s -o %t &&
// RUN: grep addrspace\(1\) %t | count 9 &&
// RUN: grep addrspace\(2\) %t | count 9

// Check that we don't lose the address space when accessing a member
// of a structure.

#define __addr1    __attribute__((address_space(1)))
#define __addr2    __attribute__((address_space(2)))

typedef struct S {
  int a;
  int b;
} S;

void test_addrspace(__addr1 S* p1, __addr2 S*p2) {
  // swap
  p1->a = p2->b;
  p1->b = p2->a;
}
