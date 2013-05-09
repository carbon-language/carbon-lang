// RUN: %clang_cc1 -fsyntax-only -verify -Wno-unused-value -std=c++11 %s

class C {

  void f() {
    int foo, bar;

    // fail to parse as a lambda introducer, so we get objc message parsing errors instead
    [foo,+] {}; // expected-error {{expected expression}}

    []; // expected-error {{expected body of lambda expression}}
    [=,foo+] {}; // expected-error {{expected ',' or ']' in lambda capture list}}
    [&this] {}; // expected-error {{cannot take the address of an rvalue of type 'C *'}}
    [] {}; 
    [=] (int i) {}; 
    [&] (int) mutable -> void {}; 
    [foo,bar] () { return 3; }; 
    [=,&foo] () {}; 
    [this] () {}; 

    [foo(bar)] () {}; // expected-error {{not supported}}
    [foo = bar] () {}; // expected-error {{not supported}}
    [foo{bar}] () {}; // expected-error {{not supported}}
    [foo = {bar}] () {}; // expected-error {{not supported}}

    [foo(bar) baz] () {}; // expected-error {{called object type 'int' is not a function}}

    // FIXME: These are some appalling diagnostics.
    [foo = bar baz]; // expected-error {{missing '['}} expected-warning 2{{receiver type 'int'}} expected-warning 2{{instance method '-baz'}}
  }

};

