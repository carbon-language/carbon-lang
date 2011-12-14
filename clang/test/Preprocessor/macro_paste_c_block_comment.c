// RUN: %clang_cc1 %s -Eonly -verify

#define COMM / ## *
COMM // expected-error {{pasting formed '/*', an invalid preprocessing token}}

