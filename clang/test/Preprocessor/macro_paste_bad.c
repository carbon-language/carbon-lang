// RUN: clang-cc -Eonly -verify %s
// pasting ""x"" and ""+"" does not give a valid preprocessing token
#define XYZ  x ## +   // expected-error {{pasting formed 'x', an invalid preprocessing token}}
XYZ

