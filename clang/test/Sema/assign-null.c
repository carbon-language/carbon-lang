// RUN: clang-cc -fsyntax-only -verify %s

#include <stddef.h>

typedef void (*hookfunc)(void *arg);
hookfunc hook;

void clear_hook() {
  hook = NULL;
}
