// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T> void f1(T*) throw(T); // expected-error{{incomplete type 'Incomplete' is not allowed in exception specification}}
struct Incomplete; // expected-note{{forward}}

void test_f1(Incomplete *incomplete_p, int *int_p) {
  f1(int_p);
  f1(incomplete_p); // expected-note{{instantiation of}}
}
