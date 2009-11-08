// RUN: clang-cc  -S %s -o %t-64.s
// RUN: clang-cc  -S %s -o %t-32.s
// RUN: true

extern "C" int printf(...);

struct S {
  S() { printf("S::S\n"); }
};

struct A {
  double x;
  A() : x(), y(), s() { printf("x = %f y = %x \n", x, y); }
  int *y;
  S s;
};

A a;

int main() {
}
