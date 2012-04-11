// RUN: %clang -cc1 -verify -std=c++11 %s

8gi///===--- recovery.cpp ---===// // expected-error {{unqualified-id}}
namespace Std { // expected-note {{here}}
  typedef int Important;
}

/ redeclare as an inline namespace // expected-error {{unqualified-id}}
inline namespace Std { // expected-error {{cannot be reopened as inline}}
  Important n;
} / end namespace Std // expected-error {{unqualified-id}}
int x;
Std::Important y;

// FIXME: Recover as if the typo correction were applied.
extenr "C" { // expected-error {{did you mean 'extern'}} expected-error {{unqualified-id}}
  void f();
}
void g() {
  z = 1; // expected-error {{undeclared}}
  f(); // expected-error {{undeclared}}
}

struct S {
  int a, b, c;
  S();
};
8S::S() : a{ 5 }, b{ 6 }, c{ 2 } { // expected-error {{unqualified-id}}
  return;
}
int k;
int l = k;

5int m = { l }, n = m; // expected-error {{unqualified-id}}

namespace N {
  int
} // expected-error {{unqualified-id}}

// FIXME: Recover as if the typo correction were applied.
strcut U { // expected-error {{did you mean 'struct'}}
} *u[3]; // expected-error {{expected ';'}}
