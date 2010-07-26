#include "prefix.h"
#include "preamble.h"
int wibble(int);

// FIXME: Turn on use of preamble files

// RUN: %clang -x c-header -o %t.pch %S/Inputs/prefix.h
// RUN: c-index-test -test-load-source-reparse 5 local -I %S/Inputs -include %t %s | FileCheck %s

// CHECK: preamble.h:1:5: FunctionDecl=bar:1:5 Extent=[1:5 - 1:13]
// CHECK: preamble.h:1:12: ParmDecl=:1:12 (Definition) Extent=[1:9 - 1:13]
// CHECK: preamble.c:3:5: FunctionDecl=wibble:3:5 Extent=[3:5 - 3:16]
// CHECK: preamble.c:3:15: ParmDecl=:3:15 (Definition) Extent=[3:12 - 3:16]
