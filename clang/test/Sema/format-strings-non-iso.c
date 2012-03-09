// RUN: %clang_cc1 -fsyntax-only -verify -std=c99 -pedantic %s

int printf(const char *restrict, ...);
int scanf(const char * restrict, ...);

void f(void) {
  char *cp;

  // The 'q' length modifier.
  printf("%qd", (long long)42); // expected-warning{{'q' length modifier is not supported by ISO C}}
  scanf("%qd", (long long *)0); // expected-warning{{'q' length modifier is not supported by ISO C}}

  // The 'm' length modifier.
  scanf("%ms", &cp); // expected-warning{{'m' length modifier is not supported by ISO C}}

  // The 'S' and 'C' conversion specifiers.
  printf("%S", L"foo"); // expected-warning{{'S' conversion specifier is not supported by ISO C}}
  printf("%C", L'x'); // expected-warning{{'C' conversion specifier is not supported by ISO C}}

  // Combining 'L' with an integer conversion specifier.
  printf("%Li", (long long)42); // expected-warning{{using length modifier 'L' with conversion specifier 'i' is not supported by ISO C}}
  printf("%Lo", (long long)42); // expected-warning{{using length modifier 'L' with conversion specifier 'o' is not supported by ISO C}}
  printf("%Lu", (long long)42); // expected-warning{{using length modifier 'L' with conversion specifier 'u' is not supported by ISO C}}
  printf("%Lx", (long long)42); // expected-warning{{using length modifier 'L' with conversion specifier 'x' is not supported by ISO C}}
  printf("%LX", (long long)42); // expected-warning{{using length modifier 'L' with conversion specifier 'X' is not supported by ISO C}}

  // Positional arguments.
  printf("%1$d", 42); // expected-warning{{positional arguments are not supported by ISO C}}
}
