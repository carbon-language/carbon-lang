; RUN: opt < %s -instcombine -S | FileCheck %s

;; Target triple for gep raising case below.
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin8"

define float @test1(float %A, float %B, i1 %C) {
EntryBlock:
  ;;  select C, 0, B + select C, A, 0 -> select C, A, B
  %cf = uitofp i1 %C to float
  %s1 = select i1 %C, float 0.000000e+00, float %B
  %s2 = select i1 %C, float %A, float 0.000000e+00
  %sum = fadd fast float %s1, %s2
  ret float %sum
; CHECK-LABEL: @test1(
; CHECK: select i1 %C, float %A, float %B
}

define float @test2(float %A, float %B, i1 %C) {
EntryBlock:
  ;;  B*(uitofp i1 C) + A*(1 - uitofp i1 C) -> select C, A, B
  %cf = uitofp i1 %C to float
  %mc = fsub fast float 1.000000e+00, %cf
  %p1 = fmul fast float %A, %mc
  %p2 = fmul fast float %B, %cf
  %s1 = fadd fast float %p2, %p1
  ret float %s1
; CHECK-LABEL: @test2(
; CHECK: select i1 %C, float %B, float %A
}

define float @test3(float %A, float %B, i1 %C) {
EntryBlock:
  ;; A*(1 - uitofp i1 C) + B*(uitofp i1 C) -> select C, A, B
  %cf = uitofp i1 %C to float
  %mc = fsub fast float 1.000000e+00, %cf
  %p1 = fmul fast float %A, %mc
  %p2 = fmul fast float %B, %cf
  %s1 = fadd fast float %p1, %p2
  ret float %s1
; CHECK-LABEL: @test3(
; CHECK: select i1 %C, float %B, float %A
}

