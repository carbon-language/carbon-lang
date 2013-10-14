// RUN: %clang_cc1 -emit-llvm -x c++ < %s
// XFAIL: *

// PR17578
struct Base {
  int a;
};
struct Derived : virtual Base
{};

void foo()
{
 int xx = __builtin_offsetof(Derived, a);
}
