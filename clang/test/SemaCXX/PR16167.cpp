// RUN: %clang_cc1 -fsyntax-only -verify %s
extern void PR16167; // expected-error {{variable has incomplete type 'void'}}
