// RUN: cp %s %t
// RUN: %clang_cc1 -pedantic -Wall -fixit %t || true
// RUN: %clang_cc1 -fsyntax-only -pedantic -Wall -Werror %t

/* This is a test of the various code modification hints that are
   provided as part of warning or extension diagnostics. All of the
   warnings will be fixed by -fixit, and the resulting file should
   compile cleanly with -Werror -pedantic. */

int printf(char const *, ...);

void test() {
  printf("%0s", (int) 123);
  printf("abc%f", "testing testing 123");
  printf("%u", (long) -12);
  printf("%+.2d", (unsigned long long) 123456);
  printf("%1d", (long double) 1.23);
  printf("%Ld", (long double) -4.56);
  printf("%1$f:%2$.*3$f:%4$.*3$f\n", 1, 2, 3, 4);
}
