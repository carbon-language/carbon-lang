// RUN: %clang_cc1 -fsyntax-only -pedantic -std=c++0x -verify %s

void foo();

void bar() { };

void wibble();

;

namespace Blah {
  void f() { };
  
  void g();
}
