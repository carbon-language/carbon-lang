// RUN: clang-cc -fsyntax-only -verify %s
// C++0x N2914.

namespace A {
  namespace B { }
}

using A::B; // expected-error{{using declaration can not refer to namespace}}
