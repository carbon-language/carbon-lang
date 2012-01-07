// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s

namespace std { class type_info; };

namespace ExplicitCapture {
  int GlobalVar; // expected-note {{declared here}}

  namespace N {
    int AmbiguousVar; // expected-note {{candidate}}
  }
  int AmbiguousVar; // expected-note {{candidate}}
  using namespace N;

  class C {
    int Member;

    static void Overload(int);
    void Overload();
    virtual C& Overload(float);

    void ExplicitCapture() {
      int foo;

      [foo, foo] () {}; // expected-error {{'foo' can appear only once}} expected-error {{not supported yet}}
      [this, this] () {}; // expected-error {{'this' can appear only once}} expected-error {{not supported yet}}
      [=, foo] () {}; // expected-error {{'&' must precede a capture when}} expected-error {{not supported yet}}
      [=, &foo] () {}; // expected-error {{not supported yet}}
      [=, this] () {}; // expected-error {{'this' cannot appear}} expected-error {{not supported yet}}
      [&, foo] () {}; // expected-error {{not supported yet}}
      [&, &foo] () {}; // expected-error {{'&' cannot precede a capture when}} expected-error {{not supported yet}}
      [&, this] () {}; // expected-error {{not supported yet}}
      [&Overload] () {}; // expected-error {{does not name a variable}} expected-error {{not supported yet}}
      [&GlobalVar] () {}; // expected-error {{does not have automatic storage duration}} expected-error {{not supported yet}}
      [&AmbiguousVar] () {} // expected-error {{reference to 'AmbiguousVar' is ambiguous}} expected-error {{not supported yet}}
      [&Globalvar] () {}; // expected-error {{use of undeclared identifier 'Globalvar'; did you mean 'GlobalVar}}
    }

    void ImplicitThisCapture() {
      [](){(void)Member;}; // expected-error {{'this' cannot be implicitly captured in this context}} expected-error {{not supported yet}}
      [&](){(void)Member;}; // expected-error {{not supported yet}}
      [this](){(void)Member;}; // expected-error {{not supported yet}}
      [this]{[this]{};};// expected-error 2 {{not supported yet}}
      []{[this]{};};// expected-error {{'this' cannot be implicitly captured in this context}} expected-error 2 {{not supported yet}}
      []{Overload(3);}; // expected-error {{not supported yet}}
      []{Overload();}; // expected-error {{'this' cannot be implicitly captured in this context}} expected-error {{not supported yet}}
      []{(void)typeid(Overload());};// expected-error {{not supported yet}}
      []{(void)typeid(Overload(.5f));};// expected-error {{'this' cannot be implicitly captured in this context}} expected-error {{not supported yet}}
    }
  };

  void f() {
    [this] () {}; // expected-error {{invalid use of 'this'}} expected-error {{not supported yet}}
  }
}
