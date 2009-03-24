// RUN: clang-cc -fsyntax-only -verify -std=c++0x %s

int && r1(int &&a);

typedef int && R;
void r2(const R a) {
  int & &&ar = a; // expected-error{{'ar' declared as a reference to a reference}}
}

