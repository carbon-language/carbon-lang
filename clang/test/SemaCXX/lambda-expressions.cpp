// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s

namespace ExplicitCapture {
  int GlobalVar; // expected-note {{declared here}}

  namespace N {
    int AmbiguousVar; // expected-note {{candidate}}
  }
  int AmbiguousVar; // expected-note {{candidate}}
  using namespace N;

  class C {
    int x;

    void f(int);
    void f() {
      int foo;

      [foo, foo] () {}; // expected-error {{'foo' can appear only once}} expected-error {{not supported yet}}
      [this, this] () {}; // expected-error {{'this' can appear only once}} expected-error {{not supported yet}}
      [=, foo] () {}; // expected-error {{'&' must precede a capture when}} expected-error {{not supported yet}}
      [=, &foo] () {}; // expected-error {{not supported yet}}
      [=, this] () {}; // expected-error {{'this' cannot appear}} expected-error {{not supported yet}}
      [&, foo] () {}; // expected-error {{not supported yet}}
      [&, &foo] () {}; // expected-error {{'&' cannot precede a capture when}} expected-error {{not supported yet}}
      [&, this] () {}; // expected-error {{not supported yet}}
      [&f] () {}; // expected-error {{does not name a variable}} expected-error {{not supported yet}}
      [&GlobalVar] () {}; // expected-error {{does not have automatic storage duration}} expected-error {{not supported yet}}
      [&AmbiguousVar] () {} // expected-error {{reference to 'AmbiguousVar' is ambiguous}} expected-error {{not supported yet}}
      [&Globalvar] () {}; // expected-error {{use of undeclared identifier 'Globalvar'; did you mean 'GlobalVar}}
    }
  };

  void f() {
    [this] () {}; // expected-error {{invalid use of 'this'}} expected-error {{not supported yet}}
  }
}
