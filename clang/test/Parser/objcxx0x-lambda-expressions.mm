// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

class C {

  void f() {
    int foo, bar;

    // fail to parse as a lambda introducer, so we get objc message parsing errors instead
    [foo,+] {}; // expected-error {{expected expression}}

    []; // expected-error {{expected body of lambda expression}}
    [=,foo+] {}; // expected-error {{expected ',' or ']' in lambda capture list}}
    [&this] {}; // expected-error {{address expression must be an lvalue}}
    [] {}; // expected-error {{lambda expressions are not supported yet}}
    [=] (int i) {}; // expected-error {{lambda expressions are not supported yet}}
    [&] (int) mutable -> void {}; // expected-error {{lambda expressions are not supported yet}}
    [foo,bar] () { return 3; }; // expected-error {{lambda expressions are not supported yet}}
    [=,&foo] () {}; // expected-error {{lambda expressions are not supported yet}}
    [this] () {}; // expected-error {{lambda expressions are not supported yet}}
  }

};

