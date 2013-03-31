; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=g5 | FileCheck %s
; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=a2 | FileCheck -check-prefix=CHECK-A2 %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define float @foo(i32 %a) nounwind {
entry:
  %x = sitofp i32 %a to float
  ret float %x

; CHECK: @foo
; CHECK: extsw [[REG:[0-9]+]], 3
; CHECK: std [[REG]],
; CHECK: lfd [[REG2:[0-9]+]],
; CHECK: fcfid [[REG3:[0-9]+]], [[REG2]]
; CHECK: frsp 1, [[REG3]]
; CHECK: blr

; CHECK-A2: @foo
; CHECK-A2: stw 3,
; CHECK-A2: lfiwax [[REG:[0-9]+]],
; CHECK-A2: fcfid [[REG2:[0-9]+]], [[REG]]
; CHECK-A2: frsp 1, [[REG2]]
; CHECK-A2: blr
}

define double @goo(i32 %a) nounwind {
entry:
  %x = sitofp i32 %a to double
  ret double %x

; CHECK: @goo
; CHECK: extsw [[REG:[0-9]+]], 3
; CHECK: std [[REG]],
; CHECK: lfd [[REG2:[0-9]+]],
; CHECK: fcfid 1, [[REG2]]
; CHECK: blr

; CHECK-A2: @goo
; CHECK-A2: stw 3,
; CHECK-A2: lfiwax [[REG:[0-9]+]],
; CHECK-A2: fcfid 1, [[REG]]
; CHECK-A2: blr
}

