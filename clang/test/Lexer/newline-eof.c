// RUN: %clang_cc1 -fsyntax-only -Wnewline-eof -verify %s
// rdar://9133072

// The following line isn't terminated, don't fix it.
void foo() {} // expected-warning{{no newline at end of file}}