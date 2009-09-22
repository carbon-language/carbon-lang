#include "truncation.c.h"

struct 

// RUN: clang-cc -fsyntax-only -code-completion-at=%s.h:4:8 -o - %s | FileCheck -check-prefix=CC1 %s &&
// CHECK-CC1: X : 1
// CHECK-NEXT-CC1: Y : 1
// RUN: clang-cc -fsyntax-only -code-completion-at=%s:3:8 -o - %s | FileCheck -check-prefix=CC2 %s &&
// CHECK-CC2: X : 1
// CHECK-CC2: Xa : 1
// CHECK-CC2: Y : 1
// RUN: true
