// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s

struct s0; // expected-note {{forward declaration}}
char ar[sizeof(s0&)]; // expected-error {{invalid application of 'sizeof' to an incomplete type}}
void test() {
  char &r = ar[0];
  static_assert(alignof(r) == 1, "bad alignment");
  static_assert(sizeof(r) == 1, "bad size");
}

void f(); 
void f(int); 
void g() { 
  sizeof(&f); // expected-error{{cannot resolve overloaded function from context}}
}
