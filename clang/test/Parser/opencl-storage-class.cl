// RUN: %clang_cc1 %s -verify -fsyntax-only

void test_storage_class_specs()
{
  static int a;    // expected-error {{OpenCL does not support the 'static' storage class specifier}}
  register int b;  // expected-error {{OpenCL does not support the 'register' storage class specifier}}
  extern int c;    // expected-error {{OpenCL does not support the 'extern' storage class specifier}}
  auto int d;      // expected-error {{OpenCL does not support the 'auto' storage class specifier}}
}
