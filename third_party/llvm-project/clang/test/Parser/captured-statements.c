// RUN: %clang_cc1 -verify %s

void test1()
{
  #pragma clang __debug captured x // expected-warning {{extra tokens at end of #pragma clang __debug captured directive}}
  {
  }
}

void test2()
{
  #pragma clang __debug captured
  int x; // expected-error {{expected '{'}}
}
