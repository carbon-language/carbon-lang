// RUN: %clang_cc1 -fsyntax-only -verify %s

struct X {
  template<typename T, typename U>
  static void f(int, int);
};

void f() {
  void (*ptr)(int, int) = &X::f<int, int>;
}
