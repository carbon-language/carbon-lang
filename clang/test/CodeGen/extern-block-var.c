// RUN: clang-cc %s -emit-llvm -o %t

int f() {
  extern int a;
  return a;
}
