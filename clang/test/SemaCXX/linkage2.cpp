// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace test1 {
  int x; // expected-note {{previous definition is here}}
  static int y; // expected-note {{previous definition is here}}
  void f() {} // expected-note {{previous definition is here}}

  extern "C" {
    extern int x; // expected-error {{declaration of 'x' has a different language linkage}}
    extern int y; // expected-error {{declaration of 'y' has a different language linkage}}
    void f(); // expected-error {{declaration of 'f' has a different language linkage}}
  }
}
