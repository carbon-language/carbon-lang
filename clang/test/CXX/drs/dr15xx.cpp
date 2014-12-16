// RUN: %clang_cc1 -std=c++98 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++1z -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

// expected-no-diagnostics

namespace dr1550 { // dr1550: yes
  int f(bool b, int n) {
    return (b ? (throw 0) : n) + (b ? n : (throw 0));
  }
}

namespace dr1560 { // dr1560: 3.5
  void f(bool b, int n) {
    (b ? throw 0 : n) = (b ? n : throw 0) = 0;
  }
  class X { X(const X&); };
  const X &get();
  const X &x = true ? get() : throw 0;
}
