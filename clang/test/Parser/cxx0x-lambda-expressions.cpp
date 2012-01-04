// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

class C {

  int f() {
    int foo, bar;

    []; // expected-error {{expected body of lambda expression}}
    [+] {}; // expected-error {{expected variable name or 'this' in lambda capture list}}
    [foo+] {}; // expected-error {{expected ',' or ']' in lambda capture list}}
    [foo,&this] {}; // expected-error {{'this' cannot be captured by reference}}
    [&this] {}; // expected-error {{'this' cannot be captured by reference}}
    [&,] {}; // expected-error {{expected variable name or 'this' in lambda capture list}}
    [=,] {}; // expected-error {{expected variable name or 'this' in lambda capture list}}
    [] {}; // expected-error {{lambda expressions are not supported yet}}
    [=] (int i) {}; // expected-error {{lambda expressions are not supported yet}}
    [&] (int) mutable -> void {}; // expected-error {{lambda expressions are not supported yet}}
    [foo,bar] () { return 3; }; // expected-error {{lambda expressions are not supported yet}}
    [=,&foo] () {}; // expected-error {{lambda expressions are not supported yet}}
    [&,foo] () {}; // expected-error {{lambda expressions are not supported yet}}
    [this] () {}; // expected-error {{lambda expressions are not supported yet}}

    return 1;
  }

};

