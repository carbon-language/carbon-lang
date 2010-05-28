// RUN: %clang_cc1 -fsyntax-only -verify %s

// Note: this is intentionally -fno-exceptions, not just accidentally
// so because that's the current -cc1 default.

// PR7243: redeclarations
namespace test0 {
  void foo() throw(int);
  void foo() throw();
}

// Overrides.
namespace test1 {
  struct A {
    virtual void foo() throw();
  };

  struct B : A {
    virtual void foo() throw(int);
  };
}

// Calls from less permissive contexts.  We don't actually do this
// check, but if we did it should also be disabled under
// -fno-exceptions.
namespace test2 {
  void foo() throw(int);
  void bar() throw() {
    foo();
  }
}

