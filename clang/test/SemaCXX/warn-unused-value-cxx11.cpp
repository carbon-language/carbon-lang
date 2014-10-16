// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify -Wunused-value %s
// expected-no-diagnostics

void f() __attribute__((const));

namespace PR18571 {
// Unevaluated contexts should not trigger unused result warnings.
template <typename T>
auto foo(T) -> decltype(f(), bool()) { // Should not warn.
  return true;
}

void g() {
  foo(1);
}
}
