// RUN: clang-cc -emit-llvm -o - %s | grep "store i32\* @x, i32\*\*"

int x;
class A {
  int& y;
  A() : y(x) {}
};
A z;

