// RUN: %clang_cc1 -fsyntax-only -triple i386-linux-gnu -verify -Wsentinel %s
// expected-no-diagnostics

#include <stddef.h>

// linux/stddef.h does something like this for cpp files:
#undef NULL
#define NULL 0

// glibc (and other) headers then define __need_NULL and rely on stddef.h
// to redefine NULL to the correct value again.
#define __need_NULL
#include <stddef.h>

// gtk headers then use __attribute__((sentinel)), which doesn't work if NULL
// is 0.
void f(const char* c, ...) __attribute__((sentinel));
void g() {
  f("", NULL);  // Shouldn't warn.
}
