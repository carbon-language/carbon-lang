// RUN: %clang_cc1 -fsyntax-only -verify %s

class S {
public:
  S (); 
};

struct D : S {
  D() : 
    b1(0), // expected-note {{previous initialization is here}}
    b2(1),
    b1(0), // expected-error {{multiple initializations given for non-static member 'b1'}}
    S(),   // expected-note {{previous initialization is here}}
    S()    // expected-error {{multiple initializations given for base 'S'}}
    {}
  int b1;
  int b2;
};

struct A {
  struct {
    int a;
    int b; 
  };
  A();
};

A::A() : a(10), b(20) { }
