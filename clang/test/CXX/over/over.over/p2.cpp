// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T> T f0(T, T);

void test_f0() {
  int (*f0a)(int, int) = f0;
  int (*f0b)(int, int) = &f0;
  int (*f0c)(int, float) = f0; // expected-error{{cannot initialize}}
  // FIXME: poor error message above!
}
