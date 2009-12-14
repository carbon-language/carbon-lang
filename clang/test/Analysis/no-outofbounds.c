// RUN: clang -cc1 -checker-cfref -analyze -analyzer-experimental-internal-checks -analyzer-store=basic -verify %s
// RUN: clang -cc1 -checker-cfref -analyze -analyzer-experimental-internal-checks -analyzer-store=region -verify %s
// XFAIL: *

//===----------------------------------------------------------------------===//
// This file tests cases where we should not flag out-of-bounds warnings.
//===----------------------------------------------------------------------===//

void f() {
  long x = 0;
  char *y = (char*) &x;
  char c = y[0] + y[1] + y[2]; // no-warning
}
