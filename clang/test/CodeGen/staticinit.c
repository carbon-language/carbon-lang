// RUN: clang -emit-llvm %s

void f() {
  static int i = 42;
}
