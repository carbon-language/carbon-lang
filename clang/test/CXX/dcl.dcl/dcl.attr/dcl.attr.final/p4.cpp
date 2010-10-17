// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s

struct B1 {
  [[final]] virtual void f(); // expected-note {{overridden virtual function is here}}
};

struct D1 : B1 {
  void f(); // expected-error {{declaration of 'f' overrides a 'final' function}}
};

struct [[final]] B2 { // expected-note {{'B2' declared here}}
};

struct D2 : B2 { // expected-error {{derivation from 'final' struct B2}}
};
