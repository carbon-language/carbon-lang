; RUN: opt < %s -S -speculative-execution \
; RUN:   -spec-exec-max-speculation-cost 4 -spec-exec-max-not-hoisted 3 \
; RUN:   | FileCheck %s

declare float @llvm.fabs.f32(float) nounwind readnone
declare i32 @llvm.ctlz.i32(i32, i1) nounwind readnone

declare float @unknown(float)
declare float @unknown_readnone(float) nounwind readnone

; CHECK-LABEL: @ifThen_fabs(
; CHECK: call float @llvm.fabs.f32(
; CHECK: br i1 true
define void @ifThen_fabs() {
  br i1 true, label %a, label %b

a:
  %x = call float @llvm.fabs.f32(float 1.0)
  br label %b

b:
  ret void
}

; CHECK-LABEL: @ifThen_ctlz(
; CHECK: call i32 @llvm.ctlz.i32(
; CHECK: br i1 true
define void @ifThen_ctlz() {
  br i1 true, label %a, label %b

a:
  %x = call i32 @llvm.ctlz.i32(i32 0, i1 true)
  br label %b

b:
  ret void
}

; CHECK-LABEL: @ifThen_call_sideeffects(
; CHECK: br i1 true
; CHECK: call float @unknown(
define void @ifThen_call_sideeffects() {
  br i1 true, label %a, label %b

a:
  %x = call float @unknown(float 1.0)
  br label %b

b:
  ret void
}

; CHECK-LABEL: @ifThen_call_readnone(
; CHECK: br i1 true
; CHECK: call float @unknown_readnone(
define void @ifThen_call_readnone() {
  br i1 true, label %a, label %b
a:
  %x = call float @unknown_readnone(float 1.0)
  br label %b

b:
  ret void
}
