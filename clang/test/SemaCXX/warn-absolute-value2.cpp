// RUN: %clang_cc1 -triple i686-pc-linux-gnu -fsyntax-only -verify %s -Wabsolute-value

extern "C" {
int abs(int);
double fabs(double);
}

using ::fabs;

double test(double x) {
  return ::abs(x);
  // expected-warning@-1{{using integer absolute value function 'abs' when argument is of floating point type}}
}
