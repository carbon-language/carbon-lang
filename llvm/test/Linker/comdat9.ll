; RUN: llvm-link %s -S -o - | FileCheck %s

$c = comdat any
@a = alias void ()* @f
define internal void @f() comdat $c {
  ret void
}

; CHECK: $c = comdat any
; CHECK: @a = alias void ()* @f
; CHECK: define internal void @f() comdat $c {
; CHECK:   ret void
; CHECK: }
