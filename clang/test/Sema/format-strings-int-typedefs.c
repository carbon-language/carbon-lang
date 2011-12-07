// RUN: %clang_cc1 -triple i386-apple-darwin9 -fsyntax-only -verify %s

int printf(char const *, ...);

void test(void) {
  printf("%jd", 42.0); // expected-warning {{conversion specifies type 'intmax_t' (aka 'long long')}}
  printf("%ju", 42.0); // expected-warning {{conversion specifies type 'uintmax_t' (aka 'unsigned long long')}}
  printf("%zu", 42.0); // expected-warning {{conversion specifies type 'size_t' (aka 'unsigned long')}}
  printf("%td", 42.0); // expected-warning {{conversion specifies type 'ptrdiff_t' (aka 'int')}}

  // typedef size_t et al. to something crazy.
  typedef void *size_t;
  typedef void *intmax_t;
  typedef void *uintmax_t;
  typedef void *ptrdiff_t;

  // The warning still fires, because it checks the underlying type.
  printf("%jd", (intmax_t)42); // expected-warning {{conversion specifies type 'intmax_t' (aka 'long long') but the argument has type 'intmax_t' (aka 'void *')}}
  printf("%ju", (uintmax_t)42); // expected-warning {{conversion specifies type 'uintmax_t' (aka 'unsigned long long') but the argument has type 'uintmax_t' (aka 'void *')}}
  printf("%zu", (size_t)42); // expected-warning {{conversion specifies type 'size_t' (aka 'unsigned long') but the argument has type 'size_t' (aka 'void *')}}
  printf("%td", (ptrdiff_t)42); // expected-warning {{conversion specifies type 'ptrdiff_t' (aka 'int') but the argument has type 'ptrdiff_t' (aka 'void *')}}
}
