; RUN: not llvm-as < %s 2>&1 | FileCheck %s

!0 = metadata !{metadata
; CHECK: expected '!' here
