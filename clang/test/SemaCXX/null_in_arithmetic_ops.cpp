// RUN: %clang_cc1 -fsyntax-only -fblocks -Wnull-arithmetic -verify %s
#include <stddef.h>

void f() {
  int a;
  bool b;

  a = 0 ? NULL + a : a + NULL; // expected-warning 2{{use of NULL in arithmetic operation}}
  a = 0 ? NULL - a : a - NULL; // expected-warning 2{{use of NULL in arithmetic operation}}
  a = 0 ? NULL / a : a / NULL; // expected-warning 2{{use of NULL in arithmetic operation}} \
                               // expected-warning {{division by zero is undefined}}
  a = 0 ? NULL * a : a * NULL; // expected-warning 2{{use of NULL in arithmetic operation}}
  a = 0 ? NULL >> a : a >> NULL; // expected-warning 2{{use of NULL in arithmetic operation}}
  a = 0 ? NULL << a : a << NULL; // expected-warning 2{{use of NULL in arithmetic operation}}
  a = 0 ? NULL % a : a % NULL; // expected-warning 2{{use of NULL in arithmetic operation}} \
                                  expected-warning {{remainder by zero is undefined}}
  a = 0 ? NULL & a : a & NULL; // expected-warning 2{{use of NULL in arithmetic operation}}
  a = 0 ? NULL | a : a | NULL; // expected-warning 2{{use of NULL in arithmetic operation}}
  a = 0 ? NULL ^ a : a ^ NULL; // expected-warning 2{{use of NULL in arithmetic operation}}

  // Using two NULLs should only give one error instead of two.
  a = NULL + NULL; // expected-warning{{use of NULL in arithmetic operation}}
  a = NULL - NULL; // expected-warning{{use of NULL in arithmetic operation}}
  a = NULL / NULL; // expected-warning{{use of NULL in arithmetic operation}} \
                   // expected-warning{{division by zero is undefined}}
  a = NULL * NULL; // expected-warning{{use of NULL in arithmetic operation}}
  a = NULL >> NULL; // expected-warning{{use of NULL in arithmetic operation}}
  a = NULL << NULL; // expected-warning{{use of NULL in arithmetic operation}}
  a = NULL % NULL; // expected-warning{{use of NULL in arithmetic operation}} \
                   // expected-warning{{remainder by zero is undefined}}
  a = NULL & NULL; // expected-warning{{use of NULL in arithmetic operation}}
  a = NULL | NULL; // expected-warning{{use of NULL in arithmetic operation}}
  a = NULL ^ NULL; // expected-warning{{use of NULL in arithmetic operation}}

  a += NULL; // expected-warning{{use of NULL in arithmetic operation}}
  a -= NULL; // expected-warning{{use of NULL in arithmetic operation}}
  a /= NULL; // expected-warning{{use of NULL in arithmetic operation}} \
             // expected-warning{{division by zero is undefined}}
  a *= NULL; // expected-warning{{use of NULL in arithmetic operation}}
  a >>= NULL; // expected-warning{{use of NULL in arithmetic operation}}
  a <<= NULL; // expected-warning{{use of NULL in arithmetic operation}}
  a %= NULL; // expected-warning{{use of NULL in arithmetic operation}} \
             // expected-warning{{remainder by zero is undefined}}
  a &= NULL; // expected-warning{{use of NULL in arithmetic operation}}
  a |= NULL; // expected-warning{{use of NULL in arithmetic operation}}
  a ^= NULL; // expected-warning{{use of NULL in arithmetic operation}}

  b = a < NULL || NULL < a; // expected-warning 2{{use of NULL in arithmetic operation}}
  b = a > NULL || NULL > a; // expected-warning 2{{use of NULL in arithmetic operation}}
  b = a <= NULL || NULL <= a; // expected-warning 2{{use of NULL in arithmetic operation}}
  b = a >= NULL || NULL >= a; // expected-warning 2{{use of NULL in arithmetic operation}}
  b = a == NULL || NULL == a; // expected-warning 2{{use of NULL in arithmetic operation}}
  b = a != NULL || NULL != a; // expected-warning 2{{use of NULL in arithmetic operation}}

  b = &a < NULL || NULL < &a || &a > NULL || NULL > &a;
  b = &a <= NULL || NULL <= &a || &a >= NULL || NULL >= &a;
  b = &a == NULL || NULL == &a || &a != NULL || NULL != &a;

  b = 0 == a;
  b = 0 == &a;

  b = ((NULL)) != a;  // expected-warning{{use of NULL in arithmetic operation}}

  void (^c)();
  b = c == NULL || NULL == c || c != NULL || NULL != c;

  class X;
  void (X::*d) ();
  b = d == NULL || NULL == d || d != NULL || NULL != d;
}
