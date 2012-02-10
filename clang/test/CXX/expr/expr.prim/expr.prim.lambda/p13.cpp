// RUN: %clang_cc1 -std=c++11 %s -Wunused -verify

void f2() {
  int i = 1;
  void g1(int = ([i]{ return i; })()); // expected-error{{lambda expression in default argument cannot capture any entity}}
  void g2(int = ([i]{ return 0; })()); // expected-error{{lambda expression in default argument cannot capture any entity}}
  void g3(int = ([=]{ return i; })()); // expected-error{{lambda expression in default argument cannot capture any entity}}
  void g4(int = ([=]{ return 0; })());
  void g5(int = ([]{ return sizeof i; })());
}
