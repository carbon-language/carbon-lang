; RUN: not llvm-as < %s 2>&1 | FileCheck %s

!0 = metadata!
; CHECK: expected '{' here
