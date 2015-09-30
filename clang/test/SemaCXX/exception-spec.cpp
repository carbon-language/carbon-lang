// RUN: %clang_cc1 -fsyntax-only -verify -fcxx-exceptions -std=c++11 %s

namespace MissingOnTemplate {
  template<typename T> void foo(T) noexcept(true); // expected-note {{previous}}
  template<typename T> void foo(T); // expected-error {{missing exception specification 'noexcept(true)'}}
  void test() { foo(0); }
}
