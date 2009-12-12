// RUN: clang-cc -fsyntax-only -verify %s
#include <stddef.h>

struct A {
  void *operator new(size_t);
};

namespace NS {
  void *operator new(size_t);; // expected-error {{'operator new' cannot be declared inside a namespace}}
}

static void *operator new(size_t); // expected-error {{'operator new' cannot be declared static in global scope}}
