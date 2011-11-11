// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c -fmodule-cache-path %t -fauto-module-import -I %S/Inputs/normal-module-map -verify %s 

#include "a1.h"
#include "b1.h"
#include "nested/nested2.h"

int test() {
  return a1 + b1 + nested2;
}
