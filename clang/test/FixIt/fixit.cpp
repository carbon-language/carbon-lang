// RUN: cp %s %t
// RUN: %clang_cc1 -pedantic -Wall -fixit -x c++ %t || true
// RUN: %clang_cc1 -fsyntax-only -pedantic -Wall -Werror -x c++ %t

/* This is a test of the various code modification hints that are
   provided as part of warning or extension diagnostics. All of the
   warnings will be fixed by -fixit, and the resulting file should
   compile cleanly with -Werror -pedantic. */

struct C1 {
  virtual void f();
  static void g();
};
struct C2 : virtual public virtual C1 { }; // expected-error{{duplicate}}

virtual void C1::f() { } // expected-error{{'virtual' can only be specified inside the class definition}}

static void C1::g() { } // expected-error{{'static' can only be specified inside the class definition}}

template<int Value> struct CT { }; // expected-note{{previous use is here}}

CT<10 >> 2> ct; // expected-warning{{require parentheses}}

class C3 {
public:
  C3(C3, int i = 0); // expected-error{{copy constructor must pass its first argument by reference}}
};

struct CT<0> { }; // expected-error{{'template<>'}}

template<> class CT<1> { }; // expected-error{{tag type}}

// Access declarations
class A {
protected:
  int foo();
};

class B : public A {
  A::foo; // expected-warning{{access declarations are deprecated}}
};

void f() throw();
void f(); // expected-warning{{missing exception specification}}

namespace rdar7853795 {
  struct A {
    bool getNumComponents() const; // expected-note{{declared here}}
    void dump() const {
      getNumComponenets(); // expected-error{{use of undeclared identifier 'getNumComponenets'; did you mean 'getNumComponents'?}}
    }
  };
}
