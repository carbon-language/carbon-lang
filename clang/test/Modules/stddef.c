// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/StdDef %s -verify -fno-modules-error-recovery

#include "ptrdiff_t.h"

ptrdiff_t pdt;

size_t st; // expected-error {{must be imported}}
// expected-note@stddef.h:* {{previous}}

#include "include_again.h"

size_t st2;
