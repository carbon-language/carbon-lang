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

namespace Test1 {
  template<typename T> struct A {};
  template<typename T> struct B : A<T> {

    B() : A<T>(), // expected-note {{previous initialization is here}} 
      A<T>() { } // expected-error {{multiple initializations given for base 'A<T>'}}
  };
}

namespace Test2 {
  template<typename T> struct A : T {
    A() : T(), // expected-note {{previous initialization is here}}
      T() { } // expected-error {{multiple initializations given for base 'T'}}
  };
}

namespace Test3 {
  template<typename T> struct A {
    T t;
    
    A() : t(1), // expected-note {{previous initialization is here}}
      t(2) { } // expected-error {{multiple initializations given for non-static member 't'}}
  };
}
