// This checks that we are not parsing module maps if modules are not enabled.

// RUN: not %clang_cc1 -fmodules -I %S/unnecessary-module-map-parsing -fsyntax-only %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -I %S/unnecessary-module-map-parsing -fsyntax-only %s

// CHECK: error: header 'unknown.h' not found

#include "a1.h"
