; The linker should choose the largest alignment when linking.

; RUN: llvm-link %p/alignment.ll %p/Inputs/alignment.ll -S | FileCheck %s
; RUN: llvm-link %p/Inputs/alignment.ll %p/alignment.ll -S | FileCheck %s

; CHECK: @X = global i32 7, align 8

@X = weak global i32 7, align 4
