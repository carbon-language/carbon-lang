// RUN: %clang_cc1 -fsyntax-only -verify %s 

void  // expected-error {{error: 'main' must return 'int'}}
main( // expected-error {{error: first parameter of 'main' (argument count) must be of type 'int'}}
     float a
) {
}
