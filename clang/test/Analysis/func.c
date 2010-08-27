// RUN: %clang_cc1 -analyze -analyzer-experimental-internal-checks -analyzer-check-objc-mem -analyzer-store=basic -verify %s
// RUN: %clang_cc1 -analyze -analyzer-experimental-internal-checks -analyzer-check-objc-mem -analyzer-store=region -verify %s

void f(void) {
  void (*p)(void);
  p = f;
  p = &f; // expected-warning{{Assigned value is always the same as the existing value}}
  p();
  (*p)();
}

void g(void (*fp)(void));

void f2() {
  g(f);
}
