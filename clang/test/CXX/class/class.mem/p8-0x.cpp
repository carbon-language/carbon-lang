// RUN: %clang_cc1 -fsyntax-only -std=c++0x -verify %s 
struct A {
  virtual void f() new new; // expected-error {{member function already marked 'new'}}
  virtual void g() override override; // expected-error {{member function already marked 'override'}}
  virtual void h() final final; // expected-error {{member function already marked 'final'}}
};

struct B {
  virtual void f() override;
  void g() override; // expected-error {{only virtual member functions can be marked 'override'}}
  int h override; // expected-error {{only virtual member functions can be marked 'override'}}
};

struct C {
  virtual void f() final;
  void g() final; // expected-error {{only virtual member functions can be marked 'final'}}
  int h final; // expected-error {{only virtual member functions can be marked 'final'}}
};
