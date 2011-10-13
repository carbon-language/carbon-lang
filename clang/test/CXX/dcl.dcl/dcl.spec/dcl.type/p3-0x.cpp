// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

using X = struct { // ok
};
template<typename T> using Y = struct { // expected-error {{can not be defined in a type alias template}}
};

class K {
  virtual ~K();
  // FIXME: Diagnostic could use some work
  operator struct S {} (); // expected-error{{ 'operator S' cannot be the name of a variable or data member}} \
  // expected-error{{expected ';' at end of declaration list}}
};

void f() {
  int arr[3] = {1,2,3};

  for (struct S { S(int) {} } s : arr) { // expected-error {{types may not be defined in a for range declaration}}
  }

  new struct T {}; // expected-error {{allocation of incomplete type}} expected-note {{forward declaration}}

  // FIXME: the diagnostic here isn't very good
  try {} catch (struct U {}); // expected-error 3{{}} expected-note 2{{}}

  (void)(struct V { V(int); })0; // expected-error {{'V' can not be defined in a type specifier}}

  (void)dynamic_cast<struct W {}*>((K*)0); // expected-error {{'W' can not be defined in a type specifier}}
  (void)static_cast<struct X {}*>(0); // expected-error {{'X' can not be defined in a type specifier}}
  (void)reinterpret_cast<struct Y {}*>(0); // expected-error {{'Y' can not be defined in a type specifier}}
  (void)const_cast<struct Z {}*>((const Z*)0); // expected-error {{'Z' can not be defined in a type specifier}}
}

void g() throw (struct Ex {}) { // expected-error {{'Ex' can not be defined in a type specifier}}
}

int alignas(struct Aa {}) x; // expected-error {{'Aa' can not be defined in a type specifier}}

int a = sizeof(struct So {}); // expected-error {{'So' can not be defined in a type specifier}}
int b = alignof(struct Ao {}); // expected-error {{'Ao' can not be defined in a type specifier}}

namespace std { struct type_info; }
const std::type_info &ti = typeid(struct Ti {}); // expected-error {{'Ti' can not be defined in a type specifier}}
