// RUN: clang-cc -fsyntax-only -verify %s 
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
    // expected-error{{destructor cannot have any parameters}}   \
    // expected-error{{destructor cannot be variadic}}
};

struct D2 {
  void ~D2() { } //                          \
  // expected-error{{destructor cannot have a return type}}  
};


struct E;

typedef E E_typedef;
struct E {
  ~E_typedef(); // expected-error{{destructor cannot be declared using a typedef 'E_typedef' (aka 'struct E') of the class name}}
};

struct F {
  (~F)(); // expected-note {{previous declaration is here}}
  ~F(); // expected-error {{destructor cannot be redeclared}}
};

~; // expected-error {{expected the class name after '~' to name a destructor}}
~undef(); // expected-error {{expected the class name after '~' to name a destructor}}
~operator+(int, int);  // expected-error {{expected the class name after '~' to name a destructor}}
~F(){} // expected-error {{destructor must be a non-static member function}}

struct G {
  ~G();
};

G::~G() { }

// <rdar://problem/6841210>
struct H {
  ~H(void) { } 
};

struct X {};

struct Y {
  ~X(); // expected-error {{expected the class name after '~' to name the enclosing class}}
};
