; RUN: opt %s -scalarizer -scalarize-load-store -S | FileCheck %s

; Test the handling of loads and stores when no data layout is available.
define void @f1(<4 x float> *%dest, <4 x float> *%src) {
; CHECK: @f1(
; CHECK: %val = load <4 x float>, <4 x float>* %src, align 4
; CHECK: %val.i0 = extractelement <4 x float> %val, i32 0
; CHECK: %add.i0 = fadd float %val.i0, %val.i0
; CHECK: %val.i1 = extractelement <4 x float> %val, i32 1
; CHECK: %add.i1 = fadd float %val.i1, %val.i1
; CHECK: %val.i2 = extractelement <4 x float> %val, i32 2
; CHECK: %add.i2 = fadd float %val.i2, %val.i2
; CHECK: %val.i3 = extractelement <4 x float> %val, i32 3
; CHECK: %add.i3 = fadd float %val.i3, %val.i3
; CHECK: %add.upto0 = insertelement <4 x float> undef, float %add.i0, i32 0
; CHECK: %add.upto1 = insertelement <4 x float> %add.upto0, float %add.i1, i32 1
; CHECK: %add.upto2 = insertelement <4 x float> %add.upto1, float %add.i2, i32 2
; CHECK: %add = insertelement <4 x float> %add.upto2, float %add.i3, i32 3
; CHECK: store <4 x float> %add, <4 x float>* %dest, align 8
; CHECK: ret void
  %val = load <4 x float> , <4 x float> *%src, align 4
  %add = fadd <4 x float> %val, %val
  store <4 x float> %add, <4 x float> *%dest, align 8
  ret void
}
