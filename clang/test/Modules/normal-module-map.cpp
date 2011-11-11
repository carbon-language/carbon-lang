// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c -fmodule-cache-path %t -fauto-module-import -I %S/Inputs/normal-module-map %s -verify

// FIXME: The expected error here is temporary, since we don't yet have the
// logic to build a module from a module map.
#include "a1.h" // expected-error{{module 'libA' not found}}
#include "b1.h"
#include "nested/nested2.h"

int test() {
  return a1 + b1 + nested2;
}
