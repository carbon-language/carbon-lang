// RUN: clang-cc %s -emit-llvm-only -verify

union x {
  int a;
  float b;
  x(float y) : b(y) {}
  x(int y) : a(y) {}
};
x a(1), b(1.0f);

