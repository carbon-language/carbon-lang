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
  A(float) : i(0) try {} // expected-error {{expected '{' or ','}}
  A(int);
  A(char);
  // FIXME: There's something very strange going on here. After the first
  // inline function-try-block, subsequent inline bodies aren't parsed anymore.
  // Valgrind is silent, though, and I can't even debug this properly.
  A() try : i(0) {} catch(...) {}
  void f() try {} catch(...) {}
};

A::A(char) : i(0) try {} // expected-error {{expected '{' or ','}}
A::A(int j) try : i(j) {} catch(...) {}
