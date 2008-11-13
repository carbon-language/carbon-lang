// RUN: clang %s -fsyntax-only -verify -pedantic

int a() {int p; *(1 ? &p : (void*)(0 && (a(),1))) = 10;} // expected-error {{not assignable}}
