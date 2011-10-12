// RUN: %clang_cc1 -fsyntax-only -std=c++98 -Wc++0x-compat -verify %s

namespace N {
  template<typename T> void f(T) {} // expected-note {{here}}
  namespace M {
    template void f<int>(int); // expected-warning {{explicit instantiation of 'N::f' must occur in namespace 'N'}}
  }
}

template<typename T> void f(T) {} // expected-note {{here}}
namespace M {
  template void f<int>(int); // expected-warning {{explicit instantiation of 'f' must occur in the global namespace}}
}

void f() {
  auto int n = 0; // expected-warning {{'auto' storage class specifier is redundant and incompatible with C++11}}
}

int n;
struct S {
  char c;
}
s = { n }, // expected-warning {{non-constant-expression cannot be narrowed from type 'int' to 'char' in initializer list in C++11}} expected-note {{explicit cast}}
t = { 1234 }; // expected-warning {{constant expression evaluates to 1234 which cannot be narrowed to type 'char' in C++11}} expected-warning {{changes value}} expected-note {{explicit cast}}
