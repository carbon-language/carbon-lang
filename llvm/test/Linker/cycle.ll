; RUN: not llvm-link %s %S/Inputs/cycle.ll 2>&1 | FileCheck %s
; RUN: not llvm-link %S/Inputs/cycle.ll %s 2>&1 | FileCheck %s

; CHECK: Linking these modules creates an alias cycle

@foo = weak global i32 0
@bar = alias i32* @foo
