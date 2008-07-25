// RUN: clang < %s -fsyntax-only -verify

// size_t coming from a system header.
#include <stddef.h>
typedef __SIZE_TYPE__ size_t;



typedef const int x; // expected-error {{previous definition is here}}
extern x a;
typedef int x;  // expected-error {{typedef redefinition with different types}}
extern x a;

