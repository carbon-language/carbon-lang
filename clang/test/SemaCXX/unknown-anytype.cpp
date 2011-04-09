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
    int y = 0 + foo; // expected-error {{no known type for 'foo'; must explicitly cast this expression to use it}}
    return foo; // expected-error {{cannot initialize}}
  }
}
