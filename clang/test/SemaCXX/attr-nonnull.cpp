// RUN: %clang_cc1 -fsyntax-only -verify %s
struct S {
  static void f(const char*, const char*) __attribute__((nonnull(1)));

  // GCC has a hidden 'this' argument in member functions, so the middle
  // argument is the one that must not be null.
  void g(const char*, const char*, const char*) __attribute__((nonnull(3)));

  void h(const char*) __attribute__((nonnull(1))); // \
      expected-error{{invalid for the implicit this argument}}
};

void test(S s) {
  s.f(0, ""); // expected-warning{{null passed}}
  s.f("", 0);
  s.g("", 0, ""); // expected-warning{{null passed}}
  s.g(0, "", 0);
}
