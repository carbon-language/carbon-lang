// RUN: %clang_cc1 %s -fsyntax-only -verify -std=c++0x

namespace Test1 {

struct A {
  virtual void f(); // expected-note {{overridden virtual function is here}}
};

struct B explicit : A {
  virtual void f(); // expected-error {{overrides function without being marked 'override'}}
};

struct C {
  virtual ~C();
};

struct D explicit : C {
  virtual ~D();
};

}

