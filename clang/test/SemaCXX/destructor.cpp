// RUN: clang -fsyntax-only -verify %s 
class A {
public:
  ~A();
};

class B {
public:
  ~B() { }
};

class C {
public:
  (~C)() { }
};

struct D {
  static void ~D(int, ...) const { } //                          \
    // expected-error{{type qualifier is not allowed on this function}} \
    // expected-error{{destructor cannot be declared 'static'}}  \
    // expected-error{{destructor cannot have a return type}}    \
    // expected-error{{destructor cannot have any parameters}}   \
    // expected-error{{destructor cannot be variadic}}
};

struct E;

typedef E E_typedef;
struct E {
  ~E_typedef(); // expected-error{{destructor cannot be declared using a typedef 'E_typedef' of the class name}}
};

struct F {
  (~F)(); // expected-error{{previous declaration is here}}
  ~F(); // expected-error{{destructor cannot be redeclared}}
};

~; // expected-error {{expected class name}}
~undef(); // expected-error {{expected class name}}
~F(){} // expected-error {{destructor must be a non-static member function}}
