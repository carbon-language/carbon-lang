// RUN: not --crash %clang_cc1 %s 2>&1 | FileCheck %s
// REQUIRES: crash-recovery

#prag\
ma clang __debug crash

// CHECK: prag\
// CHECK-NEXT: ma

