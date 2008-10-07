// RUN: clang -fsyntax-only -verify %s 

int x(1);
int (x2)(1);

void f() {
  int x(1);
  int (x2)(1); // expected-warning {{statement was disambiguated as declaration}}
  for (int x(1);;) {}
}
