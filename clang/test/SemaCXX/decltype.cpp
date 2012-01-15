// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

// PR5290
int const f0();
void f0_test() {
  decltype(0, f0()) i = 0;
  i = 0;
}

struct A { int a[1]; A() { } };
typedef A const AC;
int &f1(int*);
float &f2(int const*);

void test_f2() {
  float &fr = f2(AC().a);
}

namespace pr10154 {
  class A{
      A(decltype(nullptr) param);
  };
}

template<typename T> struct S {};
template<typename T> auto f(T t) -> decltype(S<int>(t)) {
  using U = decltype(S<int>(t));
  using U = S<int>;
  return S<int>(t);
}
