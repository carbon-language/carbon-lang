// PR15642
// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir
// RUN: echo > '%t.dir/    .h'
// RUN: echo > '%t.dir/$$.h'
// RUN: echo > '%t.dir/##.h'
// RUN: echo > '%t.dir/normal.h'
// RUN: cd %t.dir
// RUN: %clang -MD -MF - %s -fsyntax-only -I. | FileCheck -strict-whitespace %s
// RUN: %clang -MD -MF - -MV %s -fsyntax-only -I. | FileCheck -strict-whitespace %s --check-prefix=QUOTE

// CHECK: \ \ \ \ .h
// CHECK: $$$$.h
// CHECK: \#\#.h
// QUOTE: "    .h"
// QUOTE: "$$.h"
// QUOTE: "##.h"
// QUOTE-NOT: "
// QUOTE: normal.h
// QUOTE-NOT: "

#include "    .h"
#include "$$.h"
#include "##.h"
#include "normal.h"
