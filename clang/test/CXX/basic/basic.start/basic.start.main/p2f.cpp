// RUN: %clang_cc1 -fsyntax-only -verify %s 

void  // expected-error {{error: 'main' must return 'int'}}
main( // expected-error {{error: first argument of 'main' should be of type 'int'}}
     float a
) {
}
