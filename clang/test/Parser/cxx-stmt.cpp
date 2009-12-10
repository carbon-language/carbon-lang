// RUN: clang-cc -fsyntax-only -verify %s

void f()
{
  try {
    ;
  } catch(int i) {
    ;
  } catch(...) {
  }
}

void g()
{
  try; // expected-error {{expected '{'}}

  try {}
  catch; // expected-error {{expected '('}}

  try {}
  catch (...); // expected-error {{expected '{'}}

  try {}
  catch {} // expected-error {{expected '('}}
}

void h() try {
} catch(...) {
}

struct A {
  int i;
  A(int);
  A(char);
  A() try : i(0) {} catch(...) {}
  void f() try {} catch(...) {}
  A(float) : i(0) try {} // expected-error {{expected '{' or ','}}
};

A::A(char) : i(0) try {} // expected-error {{expected '{' or ','}}
A::A(int j) try : i(j) {} catch(...) {}
