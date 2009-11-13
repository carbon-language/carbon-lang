// RUN: clang-cc -analyze -analyzer-experimental-internal-checks -checker-cfref -analyzer-experimental-checks -analyzer-store=region -verify %s
#include <stdlib.h>

void f1() {
  int *p = malloc(10);
  return; // expected-warning{{Allocated memory never released. Potential memory leak.}}
}

// THIS TEST CURRENTLY FAILS.
void f1_b() {
  int *p = malloc(10);
}

void f2() {
  int *p = malloc(10);
  free(p);
  free(p); // expected-warning{{Try to free a memory block that has been released}}
}

// This case tests that storing malloc'ed memory to a static variable which is then returned
// is not leaked.  In the absence of known contracts for functions or inter-procedural analysis,
// this is a conservative answer.
int *f3() {
  static int *p = 0;
  p = malloc(10); // no-warning
  return p;
}

// This case tests that storing malloc'ed memory to a static global variable which is then returned
// is not leaked.  In the absence of known contracts for functions or inter-procedural analysis,
// this is a conservative answer.
static int *p_f4 = 0;
int *f4() {
  p_f4 = malloc(10); // no-warning
  return p_f4;
}
