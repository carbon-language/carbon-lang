// RUN: %clang_cc1 -fsyntax-only -verify %s

#include <stddef.h>

typedef void (*hookfunc)(void *arg);
hookfunc hook;

void clear_hook() {
  hook = NULL;
}
