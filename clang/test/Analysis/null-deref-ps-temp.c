// RUN: %clang_cc1 -triple i386-apple-darwin10 -analyze -analyzer-experimental-internal-checks -std=gnu99 -analyzer-check-objc-mem -analyzer-store=region -analyzer-constraints=range -analyzer-no-purge-dead -verify %s -Wreturn-type

// This is a temporary file to isolate a test case that would cause a failure
// only some of the time in null-deref-ps.c. The idempotent operations checker
// has revealed a bug on line 18 ('=' instead of '==') when the
// -analyzer-no-purge-dead flag is passed to cc1. Some fundamental design
// changes are needed to make this work without the -analyzer-no-purge-dead flag
// and this test will be integrated back into the main file when this happens.

typedef unsigned uintptr_t;

int f4_b() {
  short array[2];
  uintptr_t x = array; // expected-warning{{incompatible pointer to integer conversion}}
  short *p = x; // expected-warning{{incompatible integer to pointer conversion}}

  // The following branch should be infeasible.
  if (!(p = &array[0])) { // expected-warning{{Assigned value is always the same as the existing value}}
    p = 0;
    *p = 1; // no-warning
  }

  if (p) {
    *p = 5; // no-warning
    p = 0;
  }
  else return; // expected-warning {{non-void function 'f4_b' should return a value}}

  *p += 10; // expected-warning{{Dereference of null pointer}}
  return 0;
}
