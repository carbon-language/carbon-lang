// RUN: %clang_cc1 -triple i386-apple-darwin9 -emit-llvm -o /dev/null %s

// PR6805
void foo() {
  union { int i; } u;
  u.i=1;
}
