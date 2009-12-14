// RUN: clang -cc1 -analyze -analyzer-experimental-internal-checks -checker-cfref -analyzer-store=basic -verify %s
// RUN: clang -cc1 -analyze -analyzer-experimental-internal-checks -checker-cfref -analyzer-store=region -verify %s

void f(void) {
  void (*p)(void);
  p = f;
  p = &f;
  p();
  (*p)();
}

void g(void (*fp)(void));

void f2() {
  g(f);
}
