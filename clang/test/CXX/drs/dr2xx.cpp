// RUN: %clang_cc1 -std=c++98 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++1y %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

namespace dr200 { // dr200: dup 214
  template <class T> T f(int);
  template <class T, class U> T f(U) = delete; // expected-error 0-1{{extension}}

  void g() {
    f<int>(1);
  }
}
