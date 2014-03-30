// RUN: %clang_cc1 -std=c++98 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++1y %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

namespace dr500 { // dr500: dup 372
  class D;
  class A {
    class B;
    class C;
    friend class D;
  };
  class A::B {};
  class A::C : public A::B {};
  class D : public A::B {};
}

// expected-no-diagnostics
