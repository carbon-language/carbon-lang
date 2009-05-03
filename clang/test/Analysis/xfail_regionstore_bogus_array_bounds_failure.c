// RUN: clang-cc -checker-cfref -analyze -analyzer-store=region -verify %s
// XFAIL

// What we are seeing: Load or store into an out-of-bound memory position
// This is bogus.

void f() {
  long x = 0;
  char *y = (char*) &x;
  char c = y[0] + y[1] + y[2]; // no-warning
}
