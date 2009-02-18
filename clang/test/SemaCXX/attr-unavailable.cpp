// RUN: clang -fsyntax-only -verify %s

int &foo(int);
double &foo(double);
void foo(...) __attribute__((__unavailable__)); // expected-note{{unavailable function is declared here}}

void test_foo(short* sp) {
  int &ir = foo(1);
  double &dr = foo(1.0);
  foo(sp); // expected-error{{call to function 'foo' that has been intentionally made unavailable}}
}
