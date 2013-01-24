// RUN: %clang_cc1 -fsyntax-only -verify %s
struct S {
  S(const char *) __attribute__((nonnull(2)));

  static void f(const char*, const char*) __attribute__((nonnull(1)));

  // GCC has a hidden 'this' argument in member functions, so the middle
  // argument is the one that must not be null.
  void g(const char*, const char*, const char*) __attribute__((nonnull(3)));

  void h(const char*) __attribute__((nonnull(1))); // \
      expected-error{{invalid for the implicit this argument}}
};

void test() {
  S s(0); // expected-warning{{null passed}}

  s.f(0, ""); // expected-warning{{null passed}}
  s.f("", 0);
  s.g("", 0, ""); // expected-warning{{null passed}}
  s.g(0, "", 0);
}

namespace rdar8769025 {
  __attribute__((nonnull)) void f0(int *&p);
  __attribute__((nonnull)) void f1(int * const &p);
  __attribute__((nonnull(2))) void f2(int i, int * const &p);

  void test_f1() {
    f1(0); // expected-warning{{null passed to a callee which requires a non-null argument}}
    f2(0, 0); // expected-warning{{null passed to a callee which requires a non-null argument}}
  }
}

namespace test3 {
__attribute__((nonnull(1))) void f(void *ptr);

void g() {
  f(static_cast<char*>((void*)0));  // expected-warning{{null passed}}
  f(static_cast<char*>(0));  // expected-warning{{null passed}}
}
}

namespace test4 {
struct X {
  bool operator!=(const void *) const __attribute__((nonnull(2)));
};
bool operator==(const X&, const void *) __attribute__((nonnull(2)));

void test(const X& x) {
  (void)(x == 0);  // expected-warning{{null passed}}
  (void)(x != 0);  // expected-warning{{null passed}}
}
}
