// RUN: %clang -fsyntax-only -Xclang -verify -std=c11 %s

noreturn int f(); // expected-error 1+{{}}

#include <stdnoreturn.h>
#include <stdnoreturn.h>
#include <stdnoreturn.h>

int g();
noreturn int g();
int noreturn g();
int g();
