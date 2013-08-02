@ RUN: llvm-mc -triple armv8 -show-encoding < %s 2>&1 | FileCheck %s
setend be
@ CHECK: warning: deprecated on armv8
