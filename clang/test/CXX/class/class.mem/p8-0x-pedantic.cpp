// RUN: %clang_cc1 -fsyntax-only -std=c++0x -pedantic -verify %s 

namespace inline_extension {
  struct Base1 { 
    virtual void f() {}
  };

  struct B : Base1 {
    virtual void f() override {} // expected-warning {{'override' keyword only allowed in declarations, allowed as an extension}}
    virtual void g() final {} // expected-warning {{'final' keyword only allowed in declarations, allowed as an extension}}
    virtual void h() new {} // expected-warning {{'new' keyword only allowed in declarations, allowed as an extension}}
  };
}

