// RUN: %clang_cc1 -fsyntax-only -std=c++11 -Wc++98-compat-pedantic -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++98 -Werror %s

// -Wc++98-compat-pedantic warns on C++11 features which we accept without a
// warning in C++98 mode.

#line 32767 // ok
#line 32768 // expected-warning {{#line number greater than 32767 is incompatible with C++98}}

#define VA_MACRO(x, ...) x // expected-warning {{variadic macros are incompatible with C++98}}
VA_MACRO(,x) // expected-warning {{empty macro argument list is incompatible with C++98}}
