// RUN: %clang_cc1 -fsyntax-only -std=c++0x -verify %s 
struct A {
  virtual void f() new new; // expected-error {{member function already marked 'new'}}
  virtual void g() override override; // expected-error {{member function already marked 'override'}}
  virtual void h() final final; // expected-error {{member function already marked 'final'}}
};
