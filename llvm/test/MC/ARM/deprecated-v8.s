@ RUN: llvm-mc -triple armv8 -show-encoding < %s 2>&1 | FileCheck %s
setend be
@ CHECK: warning: deprecated
mcr p8, #0, r5, c7, c5, #4
@ CHECK: warning: deprecated on armv8
