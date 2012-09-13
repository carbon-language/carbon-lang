// RUN: %clang_cc1 -fsyntax-only -verify -triple i386-apple-darwin9 -pedantic -DALLOWED %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple thumbv6-apple-ios4.0 -pedantic -DALLOWED %s

// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64-mingw32 -pedantic %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple i686-pc-win32 -pedantic %s

// RUN: %clang_cc1 -fsyntax-only -verify -triple i686-linux-gnu -pedantic %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64-unknown-freebsd -pedantic %s

int printf(const char *restrict, ...);
int scanf(const char * restrict, ...) ;

void test() {
  int justRight = 1;
  long tooLong = 2;

  printf("%D", justRight);
  printf("%D", tooLong);
  printf("%U", justRight);
  printf("%U", tooLong);
  printf("%O", justRight);
  printf("%O", tooLong);

#ifdef ALLOWED
  // expected-warning@-8 {{'D' conversion specifier is not supported by ISO C}}
  // expected-warning@-8 {{'D' conversion specifier is not supported by ISO C}} expected-warning@-8 {{format specifies type 'int' but the argument has type 'long'}}
  // expected-warning@-8 {{'U' conversion specifier is not supported by ISO C}}
  // expected-warning@-8 {{'U' conversion specifier is not supported by ISO C}} expected-warning@-8 {{format specifies type 'unsigned int' but the argument has type 'long'}}
  // expected-warning@-8 {{'O' conversion specifier is not supported by ISO C}}
  // expected-warning@-8 {{'O' conversion specifier is not supported by ISO C}} expected-warning@-8 {{format specifies type 'unsigned int' but the argument has type 'long'}}
#else
  // expected-warning@-15 {{invalid conversion specifier 'D'}}
  // expected-warning@-15 {{invalid conversion specifier 'D'}}
  // expected-warning@-15 {{invalid conversion specifier 'U'}}
  // expected-warning@-15 {{invalid conversion specifier 'U'}}
  // expected-warning@-15 {{invalid conversion specifier 'O'}}
  // expected-warning@-15 {{invalid conversion specifier 'O'}}
#endif
}

#ifdef ALLOWED
void testPrintf(short x, long y) {
  printf("%hD", x); // expected-warning{{conversion specifier is not supported by ISO C}}
  printf("%lD", y); // expected-warning{{conversion specifier is not supported by ISO C}}
  printf("%hU", x); // expected-warning{{conversion specifier is not supported by ISO C}}
  printf("%lU", y); // expected-warning{{conversion specifier is not supported by ISO C}}
  printf("%hO", x); // expected-warning{{conversion specifier is not supported by ISO C}}
  printf("%lO", y); // expected-warning{{conversion specifier is not supported by ISO C}}

  printf("%+'0.5lD", y); // expected-warning{{conversion specifier is not supported by ISO C}}
  printf("% '0.5lD", y); // expected-warning{{conversion specifier is not supported by ISO C}}
  printf("%#0.5lO", y); // expected-warning{{conversion specifier is not supported by ISO C}}
  printf("%'0.5lU", y); // expected-warning{{conversion specifier is not supported by ISO C}}
}

void testScanf(short *x, long *y) {
  scanf("%hD", x); // expected-warning{{conversion specifier is not supported by ISO C}}
  scanf("%lD", y); // expected-warning{{conversion specifier is not supported by ISO C}}
  scanf("%hU", x); // expected-warning{{conversion specifier is not supported by ISO C}}
  scanf("%lU", y); // expected-warning{{conversion specifier is not supported by ISO C}}
  scanf("%hO", x); // expected-warning{{conversion specifier is not supported by ISO C}}
  scanf("%lO", y); // expected-warning{{conversion specifier is not supported by ISO C}}
}
#endif
