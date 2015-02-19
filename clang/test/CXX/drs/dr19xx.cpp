// RUN: %clang_cc1 -std=c++98 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++1z %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

namespace std { struct type_info; }

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

#if __cplusplus >= 201103L
namespace dr1940 { // dr1940: yes
static union {
  static_assert(true, "");  // ok
  static_assert(false, ""); // expected-error {{static_assert failed}}
};
}
#endif

#if __cplusplus >= 201402L
namespace dr1947 { // dr1947: yes
unsigned o = 0'01;  // ok
unsigned b = 0b'01; // expected-error {{invalid digit 'b' in octal constant}}
unsigned x = 0x'01; // expected-error {{invalid suffix 'x'01' on integer constant}}
}
#endif

#if __cplusplus >= 201103L
// dr1948: yes
// FIXME: This diagnostic could be improved.
void *operator new(__SIZE_TYPE__) noexcept { return nullptr; } // expected-error{{exception specification in declaration does not match previous declaration}}
#endif

#if __cplusplus >= 201103L
namespace dr1968 { // dr1968: yes
static_assert(&typeid(int) == &typeid(int), ""); // expected-error{{not an integral constant expression}}
}
#endif

// dr1994: dup 529
