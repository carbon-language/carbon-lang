// REQUIRES: shell
// PR15642
// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir
// RUN: echo > '%t.dir/    .h'
// RUN: echo > '%t.dir/$$.h'
// RUN: echo > '%t.dir/##.h'
// RUN: cd %t.dir
// RUN: %clang -MD -MF - %s -fsyntax-only -I. | FileCheck -strict-whitespace %s

// CHECK: \ \ \ \ .h
// CHECK: $$$$.h
// CHECK: \#\#.h

#include "    .h"
#include "$$.h"
#include "##.h"
