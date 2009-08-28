// RUN: clang-cc -fsyntax-only -verify %s
// C++0x N2914.

struct X {
  int i;
  static int a;
};

using X::i; // expected-error{{error: using declaration refers to class member}}
using X::s; // expected-error{{error: using declaration refers to class member}}

void f() {
  using X::i; // expected-error{{error: using declaration refers to class member}}
  using X::s; // expected-error{{error: using declaration refers to class member}}
}
