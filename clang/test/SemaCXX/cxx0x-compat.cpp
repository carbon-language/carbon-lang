// RUN: %clang_cc1 -fsyntax-only -std=c++98 -Wc++11-compat -verify %s

namespace N {
  template<typename T> void f(T) {} // expected-note 2{{here}}
  namespace M {
    template void ::N::f<int>(int); // expected-warning {{explicit instantiation of 'f' not in a namespace enclosing 'N'}}
  }
}
using namespace N;
template void f<char>(char); // expected-warning {{explicit instantiation of 'N::f' must occur in namespace 'N'}}

template<typename T> void g(T) {} // expected-note 2{{here}}
namespace M {
  template void g<int>(int); // expected-warning {{explicit instantiation of 'g' must occur at global scope}}
  template void ::g<char>(char); // expected-warning {{explicit instantiation of 'g' must occur at global scope}}
}

template inline void g<double>(double); // expected-warning {{explicit instantiation cannot be 'inline'}}

void g() {
  auto int n = 0; // expected-warning {{'auto' storage class specifier is redundant and incompatible with C++11}}
}

int n;
struct S {
  char c;
}
s = { n }, // expected-warning {{non-constant-expression cannot be narrowed from type 'int' to 'char' in initializer list in C++11}} expected-note {{explicit cast}}
t = { 1234 }; // expected-warning {{constant expression evaluates to 1234 which cannot be narrowed to type 'char' in C++11}} expected-warning {{changes value}} expected-note {{explicit cast}}
