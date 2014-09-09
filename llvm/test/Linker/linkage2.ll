; RUN: llvm-link %s %p/Inputs/linkage2.ll -S | FileCheck %s
; RUN: llvm-link %p/Inputs/linkage2.ll %s -S | FileCheck %s

@test1_a = common global i8 0

; CHECK: @test1_a = common global i8 0
