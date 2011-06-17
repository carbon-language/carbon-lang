// RUN: %clang_cc1 -fsyntax-only -verify %s
// rdar: //6734520

int foo(int)  __attribute__((__unavailable__("USE IFOO INSTEAD"))); // expected-note {{function has been explicitly marked unavailable here}}
double dfoo(double)  __attribute__((__unavailable__("NO LONGER"))); // expected-note 2 {{function has been explicitly marked unavailable here}}

void bar() __attribute__((__unavailable__)); // expected-note {{explicitly marked unavailable}}

void test_foo() {
  int ir = foo(1); // expected-error {{'foo' is unavailable: USE IFOO INSTEAD}}
  double dr = dfoo(1.0); // expected-error {{'dfoo' is unavailable: NO LONGER}}

  void (*fp)() = &bar; // expected-error {{'bar' is unavailable}}

  double (*fp4)(double) = dfoo;  // expected-error {{'dfoo' is unavailable: NO LONGER}}
}

char test2[__has_feature(attribute_unavailable_with_message) ? 1 : -1];

// rdar://9623855
void unavail(void)  __attribute__((__unavailable__));
void unavail(void) {
  // No complains inside an unavailable function.
  int ir = foo(1);
  double dr = dfoo(1.0);
  void (*fp)() = &bar;
  double (*fp4)(double) = dfoo;
}
