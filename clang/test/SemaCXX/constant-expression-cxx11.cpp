// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

template<typename T> constexpr T id(const T &t) { return t; }

struct MemberZero {
  constexpr int zero() { return 0; }
};

namespace TemplateArgumentConversion {
  template<int n> struct IntParam {};

  using IntParam0 = IntParam<0>;
  // FIXME: This should be accepted once we do constexpr function invocation.
  using IntParam0 = IntParam<id(0)>; // expected-error {{not an integral constant expression}}
  using IntParam0 = IntParam<MemberZero().zero>; // expected-error {{did you mean to call it with no arguments?}} expected-error {{not an integral constant expression}}
}

namespace CaseStatements {
  void f(int n) {
    switch (n) {
    // FIXME: Produce the 'add ()' fixit for this.
    case MemberZero().zero: // desired-error {{did you mean to call it with no arguments?}} expected-error {{not an integer constant expression}}
    // FIXME: This should be accepted once we do constexpr function invocation.
    case id(1): // expected-error {{not an integer constant expression}}
      return;
    }
  }
}
