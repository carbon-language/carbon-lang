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

struct B {
  B(decltype(undeclared)); // expected-error {{undeclared identifier}}
};
struct C {
  C(decltype(undeclared; // expected-error {{undeclared identifier}} \
                         // expected-error {{expected ')'}} expected-note {{to match this '('}}
};

namespace PR16529 {
  struct U {};
  template <typename T> struct S {
    static decltype(T{}, U{}) &f();
  };
  U &r = S<int>::f();
}

template<typename>
class conditional {
};

void foo(conditional<decltype((1),int>) {  // expected-note 2 {{to match this '('}} expected-error {{expected ')'}}
} // expected-error {{expected function body after function declarator}} expected-error 2 {{expected '>'}} expected-error {{expected ')'}}
