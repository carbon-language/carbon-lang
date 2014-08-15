; RUN: llvm-link %s %p/Inputs/comdat9.ll -S -o - | FileCheck %s

; CHECK: $c = comdat any
; CHECK: @a = alias void ()* @f
; CHECK: define internal void @f() comdat $c {
; CHECK:   ret void
; CHECK: }
