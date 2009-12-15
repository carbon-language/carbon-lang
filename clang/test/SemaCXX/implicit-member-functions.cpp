// RUN: %clang_cc1 -fsyntax-only -verify %s

struct A { }; // expected-note {{previous implicit declaration is here}}
A::A() { } // expected-error {{definition of implicitly declared constructor}}

struct B { }; // expected-note {{previous implicit declaration is here}}
B::B(const B&) { } // expected-error {{definition of implicitly declared copy constructor}}

struct C { }; // expected-note {{previous implicit declaration is here}}
C& C::operator=(const C&) { return *this; } // expected-error {{definition of implicitly declared copy assignment operator}}

struct D { }; // expected-note {{previous implicit declaration is here}}
D::~D() { } // expected-error {{definition of implicitly declared destructor}}

