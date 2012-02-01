// RUN: %clang_cc1 -std=c++11 %s -verify

class X0 {
  void explicit_capture() {
    int foo;

    [foo, foo] () {}; // expected-error {{'foo' can appear only once}} expected-error {{not supported yet}}
    [this, this] () {}; // expected-error {{'this' can appear only once}} expected-error {{not supported yet}}
    [=, foo] () {}; // expected-error {{'&' must precede a capture when}} expected-error {{not supported yet}}
    [=, &foo] () {}; // expected-error {{not supported yet}}
    [=, this] () {}; // expected-error {{'this' cannot appear}} expected-error {{not supported yet}}
    [&, foo] () {}; // expected-error {{not supported yet}}
    [&, &foo] () {}; // expected-error {{'&' cannot precede a capture when}} expected-error {{not supported yet}}
    [&, this] () {}; // expected-error {{not supported yet}}
  }
};
