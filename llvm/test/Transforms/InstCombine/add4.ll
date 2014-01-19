; RUN: opt < %s -instcombine -S | FileCheck %s

define float @test1(float %A, float %B, i1 %C) {
EntryBlock:
  ;; A*(1 - uitofp i1 C) -> select C, 0, A
  %cf = uitofp i1 %C to float
  %mc = fsub float 1.000000e+00, %cf
  %p1 = fmul fast float %A, %mc
  ret float %p1
; CHECK-LABEL: @test1(
; CHECK: select i1 %C, float -0.000000e+00, float %A
}

define float @test2(float %A, float %B, i1 %C) {
EntryBlock:
  ;; B*(uitofp i1 C) -> select C, B, 0
  %cf = uitofp i1 %C to float
  %p2 = fmul fast float %B, %cf
  ret float %p2
; CHECK-LABEL: @test2(
; CHECK: select i1 %C, float %B, float -0.000000e+00
}

define float @test3(float %A, float %B, i1 %C) {
EntryBlock:
  ;;  select C, 0, B + select C, A, 0 -> select C, A, B
  %cf = uitofp i1 %C to float
  %s1 = select i1 %C, float 0.000000e+00, float %B
  %s2 = select i1 %C, float %A, float 0.000000e+00
  %sum = fadd fast float %s1, %s2
  ret float %sum
; CHECK-LABEL: @test3(
; CHECK: select i1 %C, float %A, float %B
}

define float @test4(float %A, float %B, i1 %C) {
EntryBlock:
  ;;  B*(uitofp i1 C) + A*(1 - uitofp i1 C) -> select C, A, B
  %cf = uitofp i1 %C to float
  %mc = fsub fast float 1.000000e+00, %cf
  %p1 = fmul fast float %A, %mc
  %p2 = fmul fast float %B, %cf
  %s1 = fadd fast float %p2, %p1
  ret float %s1
; CHECK-LABEL: @test4(
; CHECK: select i1 %C, float %B, float %A
}

define float @test5(float %A, float %B, i1 %C) {
EntryBlock:
  ;; A*(1 - uitofp i1 C) + B*(uitofp i1 C) -> select C, A, B
  %cf = uitofp i1 %C to float
  %mc = fsub fast float 1.000000e+00, %cf
  %p1 = fmul fast float %A, %mc
  %p2 = fmul fast float %B, %cf
  %s1 = fadd fast float %p1, %p2
  ret float %s1
; CHECK-LABEL: @test5(
; CHECK: select i1 %C, float %B, float %A
}

; PR15952
define float @test6(float %A, float %B, i32 %C) {
  %cf = uitofp i32 %C to float
  %mc = fsub float 1.000000e+00, %cf
  %p1 = fmul fast float %A, %mc
  ret float %p1
; CHECK-LABEL: @test6(
; CHECK: uitofp
}

define float @test7(float %A, float %B, i32 %C) {
  %cf = uitofp i32 %C to float
  %p2 = fmul fast float %B, %cf
  ret float %p2
; CHECK-LABEL: @test7(
; CHECK: uitofp
}

define <4 x float> @test8(<4 x float> %A, <4 x float> %B, <4 x i1> %C) {
  ;;  B*(uitofp i1 C) + A*(1 - uitofp i1 C) -> select C, A, B
  %cf = uitofp <4 x i1> %C to <4 x float>
  %mc = fsub fast <4 x float> <float 1.0, float 1.0, float 1.0, float 1.0>, %cf
  %p1 = fmul fast <4 x float> %A, %mc
  %p2 = fmul fast <4 x float> %B, %cf
  %s1 = fadd fast <4 x float> %p2, %p1
  ret <4 x float> %s1
; CHECK-LABEL: @test8(
; CHECK: select <4 x i1> %C, <4 x float> %B, <4 x float> %A
}

define <4 x float> @test9(<4 x float> %A, <4 x float> %B, <4 x i1> %C) {
  ;; A*(1 - uitofp i1 C) + B*(uitofp i1 C) -> select C, A, B
  %cf = uitofp <4 x i1> %C to <4 x float>
  %mc = fsub fast <4 x float> <float 1.0, float 1.0, float 1.0, float 1.0>, %cf
  %p1 = fmul fast <4 x float> %A, %mc
  %p2 = fmul fast <4 x float> %B, %cf
  %s1 = fadd fast <4 x float> %p1, %p2
  ret <4 x float> %s1
; CHECK-LABEL: @test9
; CHECK: select <4 x i1> %C, <4 x float> %B, <4 x float> %A
}
