// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 -Wimplicit-fallthrough %s

void f() {
  class C {
    void f(int x) {
      switch (x) {
        case 0:
          x++;
          [[clang::fallthrough]]; // expected-no-diagnostics
        case 1:
          x++;
          break;
      }
    }
  };
}

