// RUN: clang -emit-llvm < %s 2>&1 | grep '@foo.*global.*addrspace(1)' &&
// RUN: clang -emit-llvm < %s 2>&1 | grep '@ban.*global.*addrspace(1)' &&
// RUN: clang -emit-llvm < %s 2>&1 | grep 'load.*addrspace(1)' | count 2
int foo __attribute__((address_space(1)));
int ban[10] __attribute__((address_space(1)));

int bar() { return foo; }

int baz(int i) { return ban[i]; }
