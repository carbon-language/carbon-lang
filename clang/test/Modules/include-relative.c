// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: cp -r %S/Inputs/include-relative %t/include-relative
// RUN: cd %t
// RUN: %clang_cc1 -fmodules -x c -verify -fmodules-cache-path=%t -I include-relative %s

// expected-no-diagnostics

#include "a.h"

int f() { return n; }
