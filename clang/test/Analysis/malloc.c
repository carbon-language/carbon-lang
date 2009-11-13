// RUN: clang-cc -analyze -checker-cfref -analyzer-experimental-checks -analyzer-store=region -verify %s
#include <stdlib.h>

void f1() {
  int *p = malloc(10);
  return; // expected-warning{{Allocated memory never released. Potential memory leak.}}
}

void f2() {
  int *p = malloc(10);
  free(p);
  free(p); // expected-warning{{Try to free a memory block that has been released}}
}
