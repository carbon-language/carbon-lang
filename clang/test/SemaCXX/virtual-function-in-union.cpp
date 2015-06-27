// RUN: %clang_cc1 -fsyntax-only -verify %s

union x {
  virtual void f(); // expected-error {{unions cannot have virtual functions}}
};
