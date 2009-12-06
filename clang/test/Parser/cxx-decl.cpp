// RUN: clang-cc -verify -fsyntax-only %s

int x(*g); // expected-error {{use of undeclared identifier 'g'}}
