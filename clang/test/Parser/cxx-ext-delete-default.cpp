// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s

struct A {
  A(const A&) = delete; // expected-warning {{accepted as a C++0x extension}}
  A& operator=(const A&) = delete; // expected-warning {{accepted as a C++0x extension}}
  A() = default; // expected-warning {{accepted as a C++0x extension}}
  ~A();
};

void f() = delete; // expected-warning {{accepted as a C++0x extension}}
A::~A() = default; //expected-warning {{accepted as a C++0x extension}}
