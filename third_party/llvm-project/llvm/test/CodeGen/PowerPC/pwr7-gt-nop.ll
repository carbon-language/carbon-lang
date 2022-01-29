; RUN: llc -verify-machineinstrs < %s -mcpu=pwr7 | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind
define void @foo(float* nocapture %a, float* nocapture %b, float* nocapture readonly %c, float* nocapture %d) #0 {

; CHECK-LABEL: @foo

entry:
  %0 = load float, float* %b, align 4
  store float %0, float* %a, align 4
  %1 = load float, float* %c, align 4
  store float %1, float* %b, align 4
  %2 = load float, float* %a, align 4
  store float %2, float* %d, align 4
  ret void

; CHECK: lwz [[REG1:[0-9]+]], 0(4)
; CHECK: stw [[REG1]], 0(3)
; CHECK: lwz [[REG2:[0-9]+]], 0(5)
; CHECK: stw [[REG2]], 0(4)
; CHECK: lwz [[REG3:[0-9]+]], 0(3)
; CHECK: stw [[REG3]], 0(6)
; CHECK: blr
}

attributes #0 = { nounwind }

