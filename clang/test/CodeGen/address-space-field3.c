// RUN: clang-cc -emit-llvm < %s -o %t &&
// RUN: grep addrspace\(1\) %t | count 8 &&
// RUN: grep addrspace\(2\) %t | count 8

// Check that we don't lose the address space when accessing an array element
// inside a structure.

#define __addr1    __attribute__((address_space(1)))
#define __addr2    __attribute__((address_space(2)))

typedef struct S {
  int arr[ 3 ];
} S;

void test_addrspace(__addr1 S* p1, __addr2 S*p2, int* val, int n) {
  for (int i=0; i < 3; ++i) {
    int t = val[i];
    p1->arr[i] = p2->arr[i];
  }
}
