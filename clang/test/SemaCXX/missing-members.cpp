// RUN: clang-cc -fsyntax-only -verify %s
namespace A {
  namespace B {
    class C { };
    struct S { };
    union U { };
  }
}

void f() {
  A::B::i; // expected-error {{namespace 'A::B' has no member named 'i'}}
  A::B::C::i; // expected-error {{class 'A::B::C' has no member named 'i'}}
  ::i; // expected-error {{the global scope has no member named 'i'}}
}

int A::B::i = 10; // expected-error {{namespace 'A::B' has no member named 'i'}}
int A::B::C::i = 10; // expected-error {{class 'A::B::C' has no member named 'i'}}
int A::B::S::i = 10; // expected-error {{struct 'A::B::S' has no member named 'i'}}
int A::B::U::i = 10; // expected-error {{union 'A::B::U' has no member named 'i'}}

