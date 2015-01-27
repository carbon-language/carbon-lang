// RUN: %clang_cc1 -std=c++98 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++1z %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

namespace dr1902 { // dr1902: 3.7
  struct A {};
  struct B {
    B(A);
#if __cplusplus >= 201103L
        // expected-note@-2 {{candidate}}
#endif

    B() = delete;
#if __cplusplus < 201103L
        // expected-error@-2 {{extension}}
#endif

    B(const B&) // expected-note {{deleted here}}
#if __cplusplus >= 201103L
        // expected-note@-2 {{candidate}}
#else
        // expected-error@+2 {{extension}}
#endif
        = delete;

    operator A();
  };

  extern B b1;
  B b2(b1); // expected-error {{call to deleted}}

#if __cplusplus >= 201103L
  // This is ambiguous, even though calling the B(const B&) constructor would
  // both directly and indirectly call a deleted function.
  B b({}); // expected-error {{ambiguous}}
#endif
}
