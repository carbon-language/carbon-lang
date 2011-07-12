// RUN: %clang_cc1 -fsyntax-only -verify %s

// PR10283
void min();
void min(int);

template <typename T> void max(T);

void f() {
  fin(); //expected-error {{use of undeclared identifier 'fin'; did you mean 'min'}}
  fax(0); //expected-error {{use of undeclared identifier 'fax'; did you mean 'max'}}
}
