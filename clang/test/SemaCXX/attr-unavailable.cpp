// RUN: clang-cc -fsyntax-only -verify %s

int &foo(int);
double &foo(double);
void foo(...) __attribute__((__unavailable__)); // expected-note {{candidate function}} \
// expected-note{{function has been explicitly marked unavailable here}}

void bar(...) __attribute__((__unavailable__)); // expected-note 2{{explicitly marked unavailable}}

void test_foo(short* sp) {
  int &ir = foo(1);
  double &dr = foo(1.0);
  foo(sp); // expected-error{{call to unavailable function 'foo'}}

  void (*fp)(...) = &bar; // expected-warning{{'bar' is unavailable}}
  void (*fp2)(...) = bar; // expected-warning{{'bar' is unavailable}}

  int &(*fp3)(int) = foo;
  void (*fp4)(...) = foo; // expected-warning{{'foo' is unavailable}}
}
