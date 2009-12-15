// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 -pedantic -Werror  %s
int a1[] = { 1, 3, 5 };
void f() {
  int a2[] = { 1, 3, 5 };
}
template <typename T>
void tf() {
  T t;
  // Element type may be dependent
  T a3[] = { 1, 3, 5 };
  // As might be the initializer list, value
  int a5[] = { sizeof(T) };
  // or even type.
  int a6[] = { t.get() };
}

// Allowed by GNU extension
int a4[] = {}; // expected-warning {{zero size arrays}}
