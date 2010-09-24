// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s

struct A {
  A(const A&) = delete; // expected-warning {{deleted function definition accepted as a C++0x extension}}
  A& operator=(const A&) = delete; // expected-warning {{deleted function definition accepted as a C++0x extension}}
};

void f() = delete; // expected-warning {{deleted function definition accepted as a C++0x extension}}
