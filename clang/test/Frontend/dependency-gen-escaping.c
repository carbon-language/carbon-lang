// PR15642
// RUN: %clang -M -MG %s | FileCheck -strict-whitespace %s
// RUN: %clang -M -MG -MV %s | FileCheck -strict-whitespace %s --check-prefix=NMAKE

// CHECK: \ \ \ \ .h
// CHECK: $$$$.h
// CHECK: \#\#.h
// NMAKE: "    .h"
// NMAKE: "$$.h"
// NMAKE: "##.h"
// NMAKE-NOT: "
// NMAKE: normal.h
// NMAKE-NOT: "

#include "    .h"
#include "$$.h"
#include "##.h"
#include "normal.h"
