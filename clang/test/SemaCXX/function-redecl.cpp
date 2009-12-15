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
