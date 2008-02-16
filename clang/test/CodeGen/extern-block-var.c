// RUN: clang %s -emit-llvm

int f() {
  extern int a;
  return a;
}
