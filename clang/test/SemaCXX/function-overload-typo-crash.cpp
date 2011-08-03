// RUN: %clang_cc1 -fsyntax-only -verify %s

// PR10283
void min(); //expected-note {{'min' declared here}}
void min(int);

template <typename T> void max(T); //expected-note {{'max' declared here}}

void f() {
  fin(); //expected-error {{use of undeclared identifier 'fin'; did you mean 'min'}}
  fax(0); //expected-error {{use of undeclared identifier 'fax'; did you mean 'max'}}
}
