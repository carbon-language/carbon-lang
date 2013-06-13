// RUN: %clang_cc1 -std=c++98 %s -Wdeprecated -verify
// RUN: %clang_cc1 -std=c++11 %s -Wdeprecated -verify
// RUN: %clang_cc1 -std=c++1y %s -Wdeprecated -verify

void f() throw();
void g() throw(int);
void h() throw(...);
#if __cplusplus >= 201103L
// expected-warning@-4 {{dynamic exception specifications are deprecated}} expected-note@-4 {{use 'noexcept' instead}}
// expected-warning@-4 {{dynamic exception specifications are deprecated}} expected-note@-4 {{use 'noexcept(false)' instead}}
// expected-warning@-4 {{dynamic exception specifications are deprecated}} expected-note@-4 {{use 'noexcept(false)' instead}}
#endif

void stuff() {
  register int n;
#if __cplusplus >= 201103L
  // expected-warning@-2 {{'register' storage class specifier is deprecated}}
#endif

  bool b;
  ++b; // expected-warning {{incrementing expression of type bool is deprecated}}

  // FIXME: This is ill-formed in C++11.
  char *p = "foo"; // expected-warning {{conversion from string literal to 'char *' is deprecated}}
}

struct S { int n; };
struct T : private S {
  S::n;
#if __cplusplus < 201103L
  // expected-warning@-2 {{access declarations are deprecated; use using declarations instead}}
#else
  // expected-error@-4 {{ISO C++11 does not allow access declarations; use using declarations instead}}
#endif
};
