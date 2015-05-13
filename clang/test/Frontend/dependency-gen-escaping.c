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

// Backslash followed by # or space is handled differently than GCC does,
// because GCC doesn't emit this obscure corner case the way GNU Make wants it.
// CHECK: a\b\\\#c\\\ d.h
// These combinations are just another case for NMAKE.
// NMAKE: "a\b\#c\ d.h"

#include "a\b\#c\ d.h"
