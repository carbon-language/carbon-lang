// RUN: %clang_cc1 -fsyntax-only -verify -std=c++0x %s

class C {

  int f() {
    int foo, bar;

    []; // expected-error {{expected body of lambda expression}}
    [+] {}; // expected-error {{expected variable name or 'this' in lambda capture list}}
    [foo+] {}; // expected-error {{expected ',' or ']' in lambda capture list}}
    [foo,&this] {}; // expected-error {{'this' cannot be captured by reference}}
    [&this] {}; // expected-error {{'this' cannot be captured by reference}}
    [&,] {}; // expected-error {{ expected variable name or 'this' in lambda capture list}}
    [=,] {}; // expected-error {{ expected variable name or 'this' in lambda capture list}}
    [] {};
    [=] (int i) {};
    [&] (int) mutable -> void {};
    [foo,bar] () { return 3; };
    [=,&foo] () {};
    [&,foo] () {};
    [this] () {};

    return 1;
  }

};

