// RUN: clang -cc1 -analyze -analyzer-experimental-internal-checks -checker-cfref -analyzer-store=basic -verify %s
// RUN: clang -cc1 -analyze -analyzer-experimental-internal-checks -checker-cfref -analyzer-store=region -verify %s

void foo() {
  int *p = (int*) 0x10000; // Should not crash here.
  *p = 3;
}
