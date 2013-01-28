// RUN: %clang_cc1 -E -verify %s

#define DO_PRAGMA _Pragma 
#define STR "GCC dependency \"parse.y\"")
// expected-error@+1 {{'parse.y' file not found}}
  DO_PRAGMA (STR
