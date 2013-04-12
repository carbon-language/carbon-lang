// RUN: %clang -fsyntax-only -Xclang -verify -std=c11 %s
// RUN: %clang -fsyntax-only -Xclang -verify -std=c11 -fmodules %s
// RUN: %clang -fsyntax-only -Xclang -verify -std=c11 -ffreestanding %s

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

#define __STDC_WANT_LIB_EXT1__ 1
#include <stddef.h>
#include <stdint.h>
rsize_t x = 0;

// If we are freestanding, then also check RSIZE_MAX (in a hosted implementation
// we will use the host stdint.h, which may not yet have C11 support).
#ifndef __STDC_HOSTED__
rsize_t x2 = RSIZE_MAX;
#endif

