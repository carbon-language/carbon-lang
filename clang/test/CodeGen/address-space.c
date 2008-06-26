// RUN: clang -emit-llvm < %s 2>&1 | grep '@foo.*global.*addrspace(1)' &&
// RUN: clang -emit-llvm < %s 2>&1 | grep '@ban.*global.*addrspace(1)' &&
// RUN: clang -emit-llvm < %s 2>&1 | grep 'load.*addrspace(1)' | count 2 &&
// RUN: clang -emit-llvm < %s 2>&1 | grep 'load.*addrspace(2).. @A' &&
// RUN: clang -emit-llvm < %s 2>&1 | grep 'load.*addrspace(2).. @B'

int foo __attribute__((address_space(1)));
int ban[10] __attribute__((address_space(1)));

int bar() { return foo; }

int baz(int i) { return ban[i]; }

// Both A and B point into addrspace(2).
__attribute__((address_space(2))) int *A, *B;

void test3() {
  *A = *B;
}

