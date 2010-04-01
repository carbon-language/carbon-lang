// RUN: %clang_cc1 -analyzer-check-objc-mem -analyze -analyzer-experimental-internal-checks -analyzer-store=basic -verify %s
// RUN: %clang_cc1 -analyzer-check-objc-mem -analyze -analyzer-experimental-internal-checks -analyzer-store=region -verify %s

//===----------------------------------------------------------------------===//
// This file tests cases where we should not flag out-of-bounds warnings.
//===----------------------------------------------------------------------===//

void f() {
  long x = 0;
  char *y = (char*) &x;
  char c = y[0] + y[1] + y[2]; // no-warning
  short *z = (short*) &x;
  short s = z[0] + z[1]; // no-warning
}
