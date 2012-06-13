// RUN: %clang_cc1 %s -Eonly -verify

#define COMM / ## *
COMM // expected-error {{pasting formed '/*', an invalid preprocessing token}}

// Demonstrate that an invalid preprocessing token
// doesn't swallow the rest of the file...
#error EOF // expected-error {{EOF}}
