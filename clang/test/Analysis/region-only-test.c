// RUN: clang-cc -analyze -checker-cfref -analyzer-store=region -verify %s

// Region store must be enabled for tests in this file.

// Exercise creating ElementRegion with symbolic super region.
void foo(int* p) {
  int *x;
  int a;
  if (p[0] == 1)
    x = &a;
  if (p[0] == 1)
    (void)*x; // no-warning
}

int a[10];

int *f0() {
  int *p = a+10;
  return p; // expected-warning{{Return of Pointer Value Outside of Expected Range}}
}
