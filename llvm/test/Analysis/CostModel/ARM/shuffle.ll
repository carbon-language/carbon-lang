; RUN: opt < %s  -cost-model -analyze -mtriple=thumbv7-apple-ios6.0.0 -mcpu=swift | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32-S32"
target triple = "thumbv7-apple-ios6.0.0"

; CHECK: shuffle
define void @shuffle() {


  ;; Reverse shuffles should be lowered to vrev and possibly a vext (for
  ;; quadwords)

    ; Vector values
  ; CHECK: cost of 1 {{.*}} shuffle
  %v7 = shufflevector <2 x i8> undef, <2 x i8>undef, <2 x i32> <i32 1, i32 0>
  ; CHECK: cost of 1 {{.*}} shuffle
  %v8 = shufflevector <4 x i8> undef, <4 x i8>undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  ; CHECK: cost of 1 {{.*}} shuffle
  %v9 = shufflevector <8 x i8> undef, <8 x i8>undef, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  ; CHECK: cost of 2 {{.*}} shuffle
  %v10 = shufflevector <16 x i8> undef, <16 x i8>undef, <16 x i32> <i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>

  ; CHECK: cost of 1 {{.*}} shuffle
  %v11 = shufflevector <2 x i16> undef, <2 x i16>undef, <2 x i32> <i32 1, i32 0>
  ; CHECK: cost of 1 {{.*}} shuffle
  %v12 = shufflevector <4 x i16> undef, <4 x i16>undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  ; CHECK: cost of 2 {{.*}} shuffle
  %v13 = shufflevector <8 x i16> undef, <8 x i16>undef, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>

  ; CHECK: cost of 1 {{.*}} shuffle
  %v14 = shufflevector <2 x i32> undef, <2 x i32>undef, <2 x i32> <i32 1, i32 0>
  ; CHECK: cost of 2 {{.*}} shuffle
  %v15 = shufflevector <4 x i32> undef, <4 x i32>undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>

  ; CHECK: cost of 1 {{.*}} shuffle
  %v16 = shufflevector <2 x float> undef, <2 x float>undef, <2 x i32> <i32 1, i32 0>
  ; CHECK: cost of 2 {{.*}} shuffle
  %v17 = shufflevector <4 x float> undef, <4 x float>undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>

  ret void
}
