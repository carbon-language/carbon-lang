// RUN: %clang_cc1 -std=c11 %s -verify

typedef int type;
typedef type type;
typedef int type;

void f(int N) {
  typedef int type2;
  typedef type type2;
  typedef int type2;

  typedef int vla[N]; // expected-note{{previous definition is here}}
  typedef int vla[N]; // expected-error{{typedef redefinition with different types ('int [N]' vs 'int [N]')}}
}
