// RUN: %clang_cc1 -fsyntax-only -verify -std=c++0x %s

class C {

  void f() {
    int foo, bar;

    // fail to parse as a lambda introducer, so we get objc message parsing errors instead
    [foo,+] {}; // expected-error {{expected expression}}

    []; // expected-error {{expected body of lambda expression}}
    [=,foo+] {}; // expected-error {{expected ',' or ']' in lambda capture list}}
    [&this] {}; // expected-error {{address expression must be an lvalue}}
    [] {};
    [=] (int i) {};
    [&] (int) mutable -> void {};
    // FIXME: this error occurs because we do not yet handle lambda scopes
    // properly. I did not anticipate it because I thought it was a semantic (not
    // syntactic) check.
    [foo,bar] () { return 3; }; // expected-error {{void function 'f' should not return a value}}
    [=,&foo] () {};
    [this] () {};
  }

};

