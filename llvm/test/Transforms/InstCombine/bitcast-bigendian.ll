; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; These tests are extracted from bitcast.ll.
; Verify that they also work correctly on big-endian targets.

define float @test2(<2 x float> %A, <2 x i32> %B) {
  %tmp28 = bitcast <2 x float> %A to i64  ; <i64> [#uses=2]
  %tmp23 = trunc i64 %tmp28 to i32                ; <i32> [#uses=1]
  %tmp24 = bitcast i32 %tmp23 to float            ; <float> [#uses=1]

  %tmp = bitcast <2 x i32> %B to i64
  %tmp2 = trunc i64 %tmp to i32                ; <i32> [#uses=1]
  %tmp4 = bitcast i32 %tmp2 to float            ; <float> [#uses=1]

  %add = fadd float %tmp24, %tmp4
  ret float %add

; CHECK: @test2
; CHECK-NEXT:  %tmp24 = extractelement <2 x float> %A, i32 1
; CHECK-NEXT:  bitcast <2 x i32> %B to <2 x float>
; CHECK-NEXT:  %tmp4 = extractelement <2 x float> {{.*}}, i32 1
; CHECK-NEXT:  %add = fadd float %tmp24, %tmp4
; CHECK-NEXT:  ret float %add
}

define float @test3(<2 x float> %A, <2 x i64> %B) {
  %tmp28 = bitcast <2 x float> %A to i64
  %tmp29 = lshr i64 %tmp28, 32
  %tmp23 = trunc i64 %tmp29 to i32
  %tmp24 = bitcast i32 %tmp23 to float

  %tmp = bitcast <2 x i64> %B to i128
  %tmp1 = lshr i128 %tmp, 64
  %tmp2 = trunc i128 %tmp1 to i32
  %tmp4 = bitcast i32 %tmp2 to float

  %add = fadd float %tmp24, %tmp4
  ret float %add

; CHECK: @test3
; CHECK-NEXT:  %tmp24 = extractelement <2 x float> %A, i32 0
; CHECK-NEXT:  bitcast <2 x i64> %B to <4 x float>
; CHECK-NEXT:  %tmp4 = extractelement <4 x float> {{.*}}, i32 1
; CHECK-NEXT:  %add = fadd float %tmp24, %tmp4
; CHECK-NEXT:  ret float %add
}

