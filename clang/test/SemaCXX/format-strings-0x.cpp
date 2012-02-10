// RUN: %clang_cc1 -fsyntax-only -verify -pedantic -std=c++11 %s

extern "C" {
extern int scanf(const char *restrict, ...);
extern int printf(const char *restrict, ...);
}

void f(char **sp, float *fp) {
  scanf("%as", sp); // expected-warning{{format specifies type 'float *' but the argument has type 'char **'}}

  printf("%a", 1.0);
  scanf("%afoobar", fp);
  printf(nullptr);
  printf(*sp); // expected-warning {{not a string literal}}
}
