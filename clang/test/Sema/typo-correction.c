// RUN: %clang_cc1 -fsyntax-only -verify %s
//
// This file contains typo correction tests which hit different code paths in C
// than in C++ and may exhibit different behavior as a result.

__typeof__(struct F*) var[invalid];  // expected-error-re {{use of undeclared identifier 'invalid'{{$}}}}
