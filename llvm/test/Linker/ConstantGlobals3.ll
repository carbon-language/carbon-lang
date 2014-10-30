; RUN: llvm-link %s %S/Inputs/ConstantGlobals3.ll -S | FileCheck %s
; RUN: llvm-link %S/Inputs/ConstantGlobals3.ll %s -S | FileCheck %s

; CHECK: @X = external constant [1 x i32]

@X = external global [1 x i32]
