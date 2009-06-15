// RUN: clang-cc -E %s -verify

#define DO_PRAGMA _Pragma 
DO_PRAGMA ("GCC dependency \"blahblabh\"")  // expected-error {{file not found}}

