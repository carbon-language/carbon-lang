; RUN: opt < %s -basicaa -slp-vectorizer -S  -slp-schedule-budget=16 -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

; Test if the budget for the scheduling region size works.
; We test with a reduced budget of 16 which should prevent vectorizing the loads.

declare void @unknown()

; CHECK-LABEL: @test
; CHECK: load float
; CHECK: load float
; CHECK: load float
; CHECK: load float
; CHECK: call void @unknown
define void @test(float * %a, float * %b) {
entry:
  %l0 = load float, float* %a
  %a1 = getelementptr inbounds float, float* %a, i64 1
  %l1 = load float, float* %a1
  %a2 = getelementptr inbounds float, float* %a, i64 2
  %l2 = load float, float* %a2
  %a3 = getelementptr inbounds float, float* %a, i64 3
  %l3 = load float, float* %a3

  ; some unrelated instructions inbetween to enlarge the scheduling region
  call void @unknown()
  call void @unknown()
  call void @unknown()
  call void @unknown()
  call void @unknown()
  call void @unknown()
  call void @unknown()
  call void @unknown()
  call void @unknown()
  call void @unknown()
  call void @unknown()
  call void @unknown()
  call void @unknown()
  call void @unknown()
  call void @unknown()
  call void @unknown()
  call void @unknown()
  call void @unknown()
  call void @unknown()
  call void @unknown()
  call void @unknown()
  call void @unknown()
  call void @unknown()
  call void @unknown()
  call void @unknown()
  call void @unknown()
  call void @unknown()
  call void @unknown()

  store float %l0, float* %b
  %b1 = getelementptr inbounds float, float* %b, i64 1
  store float %l1, float* %b1
  %b2 = getelementptr inbounds float, float* %b, i64 2
  store float %l2, float* %b2
  %b3 = getelementptr inbounds float, float* %b, i64 3
  store float %l3, float* %b3
  ret void
}

