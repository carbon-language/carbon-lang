; RUN: llc < %s | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "armv7-eabi"

@foo = common global i32 0                        ; <i32*> [#uses=1]

define arm_aapcs_vfpcc i32* @bar1() nounwind readnone {
entry:
; CHECK:      movw    r0, :lower16:foo
; CHECK-NEXT: movt    r0, :upper16:foo
  ret i32* @foo
}

define arm_aapcs_vfpcc void @bar2(i32 %baz) nounwind {
entry:
; CHECK:      movw    r1, :lower16:foo
; CHECK-NEXT: movt    r1, :upper16:foo
  store i32 %baz, i32* @foo, align 4
  ret void
}
