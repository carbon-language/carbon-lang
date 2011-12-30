// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11

// Implicitly-defined default constructors are constexpr if the implicit
// definition would be.
struct NonConstexpr1 { // expected-note {{here}}
  int a;
};
struct NonConstexpr2 { // expected-note {{here}}
  NonConstexpr1 nl;
};
constexpr NonConstexpr1 nc1 = NonConstexpr1(); // expected-error {{constant expression}} expected-note {{non-constexpr constructor 'NonConstexpr1'}}
constexpr NonConstexpr2 nc2 = NonConstexpr2(); // expected-error {{constant expression}} expected-note {{non-constexpr constructor 'NonConstexpr2'}}

struct Constexpr1 {};
constexpr Constexpr1 c1 = Constexpr1(); // ok
struct NonConstexpr3 : virtual Constexpr1 {};
constexpr NonConstexpr3 nc3 = NonConstexpr3(); // expected-error {{constant expression}} expected-note {{non-literal type 'const NonConstexpr3'}}

struct Constexpr2 {
  int a = 0;
};
constexpr Constexpr2 c2 = Constexpr2(); // ok

int n;
struct Member {
  Member() : a(n) {}
  constexpr Member(int&a) : a(a) {}
  int &a;
};
struct NonConstexpr4 { // expected-note {{here}}
  Member m;
};
constexpr NonConstexpr4 nc4 = NonConstexpr4(); // expected-error {{constant expression}} expected-note {{non-constexpr constructor 'NonConstexpr4'}}
struct Constexpr3 {
  constexpr Constexpr3() : m(n) {}
  Member m;
};
constexpr Constexpr3 c3 = Constexpr3(); // ok
struct Constexpr4 {
  Constexpr3 m;
};
constexpr Constexpr4 c4 = Constexpr4(); // ok


// This rule breaks some legal C++98 programs!
struct A {}; // expected-note {{here}}
struct B {
  friend A::A(); // expected-error {{non-constexpr declaration of 'A' follows constexpr declaration}}
};
