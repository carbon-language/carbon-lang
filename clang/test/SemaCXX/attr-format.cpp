// RUN: %clang_cc1 -fsyntax-only -verify %s
struct S {
  static void f(const char*, ...) __attribute__((format(printf, 1, 2)));

  // GCC has a hidden 'this' argument in member functions which is why
  // the format argument is argument 2 here.
  void g(const char*, ...) __attribute__((format(printf, 2, 3)));
};
