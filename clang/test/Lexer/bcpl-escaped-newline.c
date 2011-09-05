// RUN: %clang_cc1 -Eonly -trigraphs %s
// RUN: %clang_cc1 -Eonly -verify %s

//\
#error bar

//??/
#error qux // expected-error {{qux}}

// Trailing whitespace!
//\ 
#error quux
