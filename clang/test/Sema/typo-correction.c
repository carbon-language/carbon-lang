// RUN: %clang_cc1 -fsyntax-only -verify %s
//
// This file contains typo correction tests which hit different code paths in C
// than in C++ and may exhibit different behavior as a result.

__typeof__(struct F*) var[invalid];  // expected-error-re {{use of undeclared identifier 'invalid'{{$}}}}

void PR21656() {
  float x;
  x = (float)arst;  // expected-error-re {{use of undeclared identifier 'arst'{{$}}}}
}

a = b ? : 0;  // expected-warning {{type specifier missing, defaults to 'int'}} \
              // expected-error {{use of undeclared identifier 'b'}}
