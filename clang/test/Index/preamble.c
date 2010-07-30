#include "prefix.h"
#include "preamble.h"
int wibble(int);

// RUN: %clang -x c-header -o %t.pch %S/Inputs/prefix.h
// RUN: env CINDEXTEST_EDITING=1 c-index-test -test-load-source-reparse 5 local -I %S/Inputs -include %t %s | FileCheck %s
// CHECK: preamble.c:3:5: FunctionDecl=wibble:3:5 Extent=[3:5 - 3:16]
// CHECK: preamble.c:3:15: ParmDecl=:3:15 (Definition) Extent=[3:12 - 3:16]
