// RUN: %clang_cc1 -analyze -analyzer-checker=core,unix.Malloc -analyzer-store region -verify %s
// XFAIL: *

void f1() {
  int *n = new int;
  if (*n) { // expected-warning {{Branch condition evaluates to a garbage value}}
  }
}

void f2() {
  int *n = new int(3);
  if (*n) { // no-warning
  }
}

void *operator new(size_t, void *, void *);
void *testCustomNew() {
  int *x = (int *)malloc(sizeof(int));  
  void *y = new (0, x) int;  
  return y; // no-warning (placement new could have freed x)
}
