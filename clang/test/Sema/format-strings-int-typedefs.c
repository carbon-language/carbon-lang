// RUN: %clang_cc1 -triple i386-apple-darwin9 -fsyntax-only -verify %s

int printf(char const *, ...);

void test(void) {
  // size_t, et al. have not been declared yet,
  // so the warning should refer to the builtin types.
  printf("%jd", 42.0); // expected-warning {{conversion specifies type 'long long'}}
  printf("%ju", 42.0); // expected-warning {{conversion specifies type 'unsigned long long'}}
  printf("%zu", 42.0); // expected-warning {{conversion specifies type 'unsigned long'}}
  printf("%td", 42.0); // expected-warning {{conversion specifies type 'int'}}

  typedef __typeof(sizeof(int)) size_t;
  typedef __INTMAX_TYPE__ intmax_t;
  typedef __UINTMAX_TYPE__ uintmax_t;
  typedef __PTRDIFF_TYPE__ ptrdiff_t;

  printf("%jd", 42.0); // expected-warning {{conversion specifies type 'intmax_t' (aka 'long long')}}
  printf("%ju", 42.0); // expected-warning {{conversion specifies type 'uintmax_t' (aka 'unsigned long long')}}
  printf("%zu", 42.0); // expected-warning {{conversion specifies type 'size_t' (aka 'unsigned long')}}
  printf("%td", 42.0); // expected-warning {{conversion specifies type 'ptrdiff_t' (aka 'int')}}
}

void test2(void) {
  typedef void *size_t;

  // The typedef for size_t does not match the builtin type,
  // so the warning should not refer to it.
  printf("%zu", 42.0); // expected-warning {{conversion specifies type 'unsigned long'}}
}
