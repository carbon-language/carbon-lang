// RUN: %clang_cc1 -fsyntax-only -Wno-deprecated-writable-strings -verify %s
// RUN: %clang_cc1 -fsyntax-only -fwritable-strings -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wno-write-strings -verify %s
// RUN: %clang_cc1 -fsyntax-only -Werror=c++11-compat -verify %s -DERROR
// rdar://8827606

char *fun(void)
{
   return "foo";
#ifdef ERROR
   // expected-error@-2 {{deprecated}}
#endif
}

void test(bool b)
{
  ++b; // expected-warning {{incrementing expression of type bool is deprecated}}
}
