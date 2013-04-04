// RUN: %clang -fsyntax-only -Xclang -verify -std=c11 %s
// RUN: %clang -fsyntax-only -Xclang -verify -std=c11 -fmodules %s

noreturn int f(); // expected-error 1+{{}}

#include <stdnoreturn.h>
#include <stdnoreturn.h>
#include <stdnoreturn.h>

int g();
noreturn int g();
int noreturn g();
int g();

#include <stdalign.h>
_Static_assert(__alignas_is_defined, "");
_Static_assert(__alignof_is_defined, "");
alignas(alignof(int)) char c[4];
_Static_assert(__alignof(c) == 4, "");
