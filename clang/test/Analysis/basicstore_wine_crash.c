// RUN: clang-cc -checker-cfref -analyze -analyzer-store=basic %s

// Once xfail_regionstore_wine_crash.c passes, move this test case
// into misc-ps.m.

void foo() {
  long x = 0;
  char *y = (char *) &x;
  if (!*y)
    return;
}
