@ RUN: llvm-mc -triple armv8 -show-encoding < %s 2>&1 | FileCheck %s --check-prefix=CHECK-V8
@ RUN: llvm-mc -triple armv7 -show-encoding < %s 2>&1 | FileCheck %s --check-prefix=CHECK-V7
setend be
@ CHECK-V8: warning: deprecated
@ CHECK-V7-NOT: warning: deprecated
mcr p8, #0, r5, c7, c5, #4
@ CHECK-V8: warning: deprecated on armv8
@ CHECK-V7-NOT: warning: deprecated on armv8
