// RUN: clang-cc %s -verify -fsyntax-only

int a __attribute__((nodebug)); // expected-warning {{'nodebug' attribute only applies to function types}}

void t1() __attribute__((nodebug));

void t2() __attribute__((nodebug(2))); // expected-error {{attribute requires 0 argument(s)}}

