// RUN: %clang_cc1 -fcatch-undefined-behavior -emit-llvm-only %s

// PR6805
void foo() {
  union { int i; } u;
  u.i=1;
}
