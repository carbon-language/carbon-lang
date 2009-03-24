// RUN: clang-cc -fsyntax-only -verify %s

// size_t coming from a system header.
#include <stddef.h>
typedef __SIZE_TYPE__ size_t;



typedef const int x; // expected-note {{previous definition is here}}
extern x a;
typedef int x;  // expected-error {{typedef redefinition with different types}}
extern x a;

// <rdar://problem/6097585>
int y; // expected-note 2 {{previous definition is here}}
float y; // expected-error{{redefinition of 'y' with a different type}}
double y; // expected-error{{redefinition of 'y' with a different type}}
