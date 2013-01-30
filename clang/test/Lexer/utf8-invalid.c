// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -E %s -o /dev/null

// Note: this file contains invalid UTF-8 before the variable name in the
// next line. Please do not fix!

extern int ‚x; // expected-error{{source file is not valid UTF-8}}

#if 0
// Don't warn about bad UTF-8 in raw lexing mode.
extern int ‚x;
#endif
