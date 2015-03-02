// RUN: cd %S
// RUN: %clang_cc1 -I. -fmodule-maps -fmodule-name=A  -fmodule-map-file=%S/Inputs/filename/module.map %s -E | FileCheck %s

#include "Inputs/filename/a.h"

// Make sure that headers that are referenced by module maps have __FILE__
// reflect the include path they were found with.
// CHECK: const char *p = "./Inputs/filename/a.h"
