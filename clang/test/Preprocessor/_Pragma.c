// RUN: clang-cc %s -E -verify

_Pragma ("GCC system_header")  // expected-warning {{system_header ignored in main file}}

