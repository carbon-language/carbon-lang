// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++0x
void f() {
  auto a = a; // expected-error{{variable 'a' declared with 'auto' type cannot appear in its own initializer}}
}

struct S { auto a; }; // expected-error{{'auto' not allowed in struct member}}

void f(auto a) // expected-error{{'auto' not allowed in function prototype}}
{
  try { } catch (auto a) {  } // expected-error{{'auto' not allowed in exception declaration}}
}

template <auto a = 10> class C { }; // expected-error{{'auto' not allowed in template parameter}}
