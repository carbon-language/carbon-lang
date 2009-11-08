// RUN: clang-cc -analyze -checker-cfref -analyzer-store=basic -verify %s
// RUN: clang-cc -analyze -checker-cfref -analyzer-store=region -verify %s

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
