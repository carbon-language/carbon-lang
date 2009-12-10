// RUN: clang-cc -emit-llvm-only -verify %s

enum A { a } __attribute((packed));
int func(A x) { return x==a; }
