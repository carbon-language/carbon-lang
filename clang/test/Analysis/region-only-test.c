// RUN: clang-cc -analyze -checker-simple -analyzer-store=region -verify %s

// Region store must be enabled for tests in this file.

// Exercise creating ElementRegion with symbolic super region.
void foo(int* p) {
  int *x;
  int a;
  if (p[0] == 1)
    x = &a;
  if (p[0] == 1)
    *x; // no-warning
}
