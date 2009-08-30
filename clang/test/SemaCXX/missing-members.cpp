// RUN: clang-cc -fsyntax-only -verify %s
namespace A {
  namespace B {
    class C { };
    struct S { };
    union U { };
  }
}

void f() {
  A::B::i; // expected-error {{no member named 'i' in namespace 'A::B'}}
  A::B::C::i; // expected-error {{no member named 'i' in class 'A::B::C'}}
  ::i; // expected-error {{no member named 'i' in the global namespace}}
}

int A::B::i = 10; // expected-error {{no member named 'i' in namespace 'A::B'}}
int A::B::C::i = 10; // expected-error {{no member named 'i' in class 'A::B::C'}}
int A::B::S::i = 10; // expected-error {{no member named 'i' in struct 'A::B::S'}}
int A::B::U::i = 10; // expected-error {{no member named 'i' in union 'A::B::U'}}

using A::B::D; // expected-error {{no member named 'D' in namespace 'A::B'}}

struct S : A::B::C { 
  using A::B::C::f; // expected-error {{no member named 'f' in class 'A::B::C'}}
  
};
