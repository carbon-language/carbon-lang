// RUN:  %clang_cc1 -std=c++14 -fconcepts-ts -x c++ -verify %s

namespace A {
  template<typename T> concept bool C1() { return true; }

  template<typename T> concept bool C2 = true;
}

template<typename T> concept bool D1(); // expected-error {{function concept declaration must be a definition}}

struct B {
  template<typename T> concept bool D2() { return true; } // expected-error {{concept declarations may only appear in namespace scope}}
};

struct C {
  template<typename T> static concept bool D3 = true; // expected-error {{concept declarations may only appear in namespace scope}}
};

concept bool D4() { return true; } // expected-error {{'concept' can only appear on the definition of a function template or variable template}}

concept bool D5 = true; // expected-error {{'concept' can only appear on the definition of a function template or variable template}}

template<typename T>
concept bool D6; // expected-error {{variable concept declaration must be initialized}}

