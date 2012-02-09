// RUN: %clang_cc1 -std=c++11 %s -verify

class X0 {
  void explicit_capture() {
    int foo;

    (void)[foo, foo] () {}; // expected-error {{'foo' can appear only once}} expected-error {{not supported yet}}
    (void)[this, this] () {}; // expected-error {{'this' can appear only once}} expected-error {{not supported yet}}
    (void)[=, foo] () {}; // expected-error {{'&' must precede a capture when}} expected-error {{not supported yet}}
    (void)[=, &foo] () {}; // expected-error {{not supported yet}}
    (void)[=, this] () {}; // expected-error {{'this' cannot appear}} expected-error {{not supported yet}}
    (void)[&, foo] () {}; // expected-error {{not supported yet}}
    (void)[&, &foo] () {}; // expected-error {{'&' cannot precede a capture when}} expected-error {{not supported yet}}
    (void)[&, this] () {}; // expected-error {{not supported yet}}
  }
};
