// RUN: %clang_cc1 -funknown-anytype -fsyntax-only -verify %s

namespace test0 {
  extern __unknown_anytype test0;
  extern __unknown_anytype test1();
  extern __unknown_anytype test2(int);
}

namespace test1 {
  extern __unknown_anytype foo;
  int test() {
    // TODO: it would be great if the 'cannot initialize' errors
    // turned into something more interesting.  It's just a matter of
    // making sure that these locations check for placeholder types
    // properly.

    int x = foo; // expected-error {{cannot initialize}}
    int y = 0 + foo; // expected-error {{'foo' has unknown type}}
    return foo; // expected-error {{cannot initialize}}
  }
}

namespace test2 {
  extern __unknown_anytype foo();
  void test() {
    foo(); // expected-error {{'foo' has unknown return type}}
  }
}

namespace test3 {
  extern __unknown_anytype foo;
  void test() {
    foo(); // expected-error {{call to unsupported expression with unknown type}}
    ((void(void)) foo)(); // expected-error {{variable 'foo' with unknown type cannot be given a function type}}
  }
}
