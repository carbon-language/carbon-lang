// RUN: %clang_cc1 -fsyntax-only -verify %s
int foo(int);

namespace N {
  void f1() {
    void foo(int); // okay
  }

  // FIXME: we shouldn't even need this declaration to detect errors
  // below.
  void foo(int); // expected-note{{previous declaration is here}}

  void f2() {
    int foo(int); // expected-error{{functions that differ only in their return type cannot be overloaded}}

    {
      int foo;
      {
        // FIXME: should diagnose this because it's incompatible with
        // N::foo. However, name lookup isn't properly "skipping" the
        // "int foo" above.
        float foo(int); 
      }
    }
  }
}

class A {
 void typocorrection(); // expected-note {{'typocorrection' declared here}}
};

void A::Notypocorrection() { // expected-error {{out-of-line definition of 'Notypocorrection' does not match any declaration in 'A'; did you mean 'typocorrection'}}
}


namespace test0 {
  void dummy() {
    void Bar(); // expected-note {{'Bar' declared here}}
    class A {
      friend void bar(); // expected-error {{no matching function 'bar' found in local scope; did you mean 'Bar'}}
    };
  }
}


class B {
 void typocorrection(const int); // expected-note {{type of 1st parameter of member declaration does not match definition}}
 void typocorrection(double);
};

void B::Notypocorrection(int) { // expected-error {{out-of-line definition of 'Notypocorrection' does not match any declaration in 'B'; did you mean 'typocorrection'}}
}

struct X { int f(); };
struct Y : public X {};
int Y::f() { return 3; } // expected-error {{out-of-line definition of 'f' does not match any declaration in 'Y'}}

namespace test1 {
struct Foo {
  class Inner { };
};
}

class Bar {
  void f(test1::Foo::Inner foo) const; // expected-note {{member declaration nearly matches}}
};

using test1::Foo;

void Bar::f(Foo::Inner foo) { // expected-error {{out-of-line definition of 'f' does not match any declaration in 'Bar'}}
  (void)foo;
}
