; RUN: opt -S -instcombine < %s | FileCheck %s
; <rdar://problem/8285027>

; CHECK: @func
define i32 @func(i32 %a) nounwind ssp align 2 {
entry:
; CHECK: and i32 %a, -256
  %0 = or i32 %a, 8
; CHECK: or i32 %0, 8
  %1 = and i32 %0, -248
  ret i32 %1
; CHECK: }
}
