// RUN: clang -fsyntax-only -verify %s 
namespace A {
  struct C {
    static int cx;
  };
  int ax;
  void Af();
}

A:: ; // expected-error {{expected unqualified-id}}
::A::ax::undef ex3; // expected-error {{expected a class or namespace}} expected-error {{expected '=', ',', ';', 'asm', or '__attribute__' after declarator}}
A::undef1::undef2 ex4; // expected-error {{no member named 'undef1'}} expected-error {{expected '=', ',', ';', 'asm', or '__attribute__' after declarator}}

class C2 {
  void m();
  int x;
};

void C2::m() {
  x = 0;
}

namespace B {
  void ::A::Af() {} // expected-error {{definition or redeclaration of 'Af' not in a namespace enclosing 'A'}}
}

void f1() {
  void A::Af(); // expected-error {{definition or redeclaration of 'Af' not allowed inside a function}}  
}

void f2() {
  A:: ; // expected-error {{expected unqualified-id}}
  A::C::undef = 0; // expected-error {{no member named 'undef'}}
  ::A::C::cx = 0;
  int x = ::A::ax = A::C::cx;
  x = sizeof(A::C);
  x = sizeof(::A::C::cx);
}

A::C c1;
struct A::C c2;
struct S : public A::C {};
struct A::undef; // expected-error {{'undef' does not name a tag member in the specified scope}}

namespace A2 {
  typedef int INT;
  struct RC;
  struct CC {
    struct NC;
  };
}

struct A2::RC {
  INT x;
};

struct A2::CC::NC {
  void m() {}
};

void f3() {
  N::x = 0; // expected-error {{use of undeclared identifier 'N'}}
  int N;
  N::x = 0; // expected-error {{expected a class or namespace}}
  { int A;           A::ax = 0; }
  { enum A {};       A::ax = 0; }
  { enum A { A };    A::ax = 0; }
  { typedef int A;   A::ax = 0; }
  { typedef int A(); A::ax = 0; }
  { typedef A::C A;  A::ax = 0; } // expected-error {{no member named 'ax'}}
  { typedef A::C A;  A::cx = 0; }
}

// make sure the following doesn't hit any asserts
void f4(undef::C); // expected-error {{use of undeclared identifier 'undef'}} // expected-error {{expected ')'}} expected-note {{to match this '('}} // expected-error {{variable has incomplete type 'void'}}
