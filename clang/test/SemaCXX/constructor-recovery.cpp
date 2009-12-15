// RUN: %clang_cc1 -fsyntax-only -verify %s

struct C {  // expected-note {{candidate function}}
  virtual C() = 0; // expected-error{{constructor cannot be declared 'virtual'}} \
                      expected-note {{candidate function}}
};

void f() {
 C c;  // expected-error {{call to constructor of 'c' is ambiguous}}
}
