// RUN: %clang_cc1 -fsyntax-only -verify %s

// Note: this file contains invalid UTF-8 before the variable name in the
// next line. Please do not fix!

extern int ‚x; // expected-error{{source file is not valid UTF-8}}
