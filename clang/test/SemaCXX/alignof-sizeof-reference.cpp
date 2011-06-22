// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s

struct s0; // expected-note {{forward declaration}}
char ar[sizeof(s0&)]; // expected-error {{invalid application of 'sizeof' to an incomplete type}}
void test() {
  char &r = ar[0];
  static_assert(alignof(r) == 1, "bad alignment");
  static_assert(sizeof(r) == 1, "bad size");
}

void f();  // expected-note{{candidate function}}
void f(int);  // expected-note{{candidate function}}
void g() { 
  sizeof(&f); // expected-error{{cannot resolve overloaded function 'f' from context}}
}

template<typename T> void f_template(); // expected-note{{candidate function}}
template<typename T> void f_template(T*); // expected-note{{candidate function}}
void rdar9659191() {
  (void)alignof(f_template<int>); // expected-error{{cannot resolve overloaded function 'f_template' from context}}
}
