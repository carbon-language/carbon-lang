// RUN: cp %s %t
// RUN: %clang_cc1 -pedantic -Wall -fixit %t || true
// RUN: %clang_cc1 -fsyntax-only -pedantic -Wall -Werror %t

/* This is a test of the various code modification hints that are
   provided as part of warning or extension diagnostics. All of the
   warnings will be fixed by -fixit, and the resulting file should
   compile cleanly with -Werror -pedantic. */

int printf(char const *, ...);

void test() {
  // Basic types
  printf("%s", (int) 123);
  printf("abc%0f", "testing testing 123");
  printf("%u", (long) -12);
  printf("%p", 123);
  printf("%c\n", "x");
  printf("%c\n", 1.23);

  // Larger types
  printf("%+.2d", (unsigned long long) 123456);
  printf("%1d", (long double) 1.23);

  // Flag handling
  printf("%0+s", (unsigned) 31337); // 0 flag should stay
  printf("%#p", (void *) 0);
  printf("% +f", 1.23); // + flag should stay
  printf("%0-f", 1.23); // - flag should stay

  // Positional arguments
  printf("%1$f:%2$.*3$f:%4$.*3$f\n", 1, 2, 3, 4);

  // Precision
  printf("%10.5d", 1l); // (bug 7394)
  printf("%.2c", 'a');

  // Ignored flags
  printf("%0-f", 1.23);

  // Bad length modifiers
  printf("%hhs", "foo");
  printf("%1$zp", (void *)0);
}
