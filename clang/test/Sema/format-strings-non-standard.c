// RUN: %clang_cc1 -fsyntax-only -verify -std=c99 -pedantic %s

int printf(const char *restrict, ...);
int scanf(const char * restrict, ...);

void f(void) {
  char *cp;

  // The 'q' length modifier.
  printf("%qd", (long long)42); // expected-warning{{'q' is a non-standard length modifier}}
  scanf("%qd", (long long *)0); // expected-warning{{'q' is a non-standard length modifier}}

  // The 'm' length modifier.
  scanf("%ms", &cp); // expected-warning{{'m' is a non-standard length modifier}}

  // The 'S' and 'C' conversion specifiers.
  printf("%S", L"foo"); // expected-warning{{'S' is a non-standard conversion specifier}}
  printf("%C", L'x'); // expected-warning{{'C' is a non-standard conversion specifier}}

  // Combining 'L' with an integer conversion specifier.
  printf("%Li", (long long)42); // expected-warning{{using the length modifier 'L' with the conversion specifier 'i' is non-standard}}
  printf("%Lo", (long long)42); // expected-warning{{using the length modifier 'L' with the conversion specifier 'o' is non-standard}}
  printf("%Lu", (long long)42); // expected-warning{{using the length modifier 'L' with the conversion specifier 'u' is non-standard}}
  printf("%Lx", (long long)42); // expected-warning{{using the length modifier 'L' with the conversion specifier 'x' is non-standard}}
  printf("%LX", (long long)42); // expected-warning{{using the length modifier 'L' with the conversion specifier 'X' is non-standard}}
}
