// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s

// Simple parser tests, dynamic specification.

namespace dyn {

  struct X { };

  struct Y { };

  void f() throw() { }

  void g(int) throw(X) { }

  void h() throw(X, Y) { }

  class Class {
    void foo() throw (X, Y) { }
  };

  void (*fptr)() throw();

}

// Simple parser tests, noexcept specification.

namespace noex {

  void f() noexcept { }
  void g() noexcept (true) { }
  void h() noexcept (false) { }
  void i() noexcept (1 < 2) { }

  class Class {
    void foo() noexcept { }
    void bar() noexcept (true) { }
  };

  void (*fptr)() noexcept;
  void (*gptr)() noexcept (true);

}

namespace bad {

  void f() throw(int) noexcept { } // expected-error {{cannot have both}}
  void g() noexcept throw(int) { } // expected-error {{cannot have both}}

}
