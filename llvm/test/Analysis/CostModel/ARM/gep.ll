; RUN: opt -cost-model -analyze -mtriple=thumbv7-apple-ios6.0.0 -mcpu=swift < %s | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32-S32"
target triple = "thumbv7-apple-ios6.0.0"

define void @test_geps() {
  ; Cost of scalar integer geps should be one. We can't always expect it to be
  ; folded into the instruction addressing mode.
;CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds i8*
  %a0 = getelementptr inbounds i8* undef, i32 0
;CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds i16*
  %a1 = getelementptr inbounds i16* undef, i32 0
;CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds i32*
  %a2 = getelementptr inbounds i32* undef, i32 0

;CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds i64*
  %a3 = getelementptr inbounds i64* undef, i32 0

  ; Cost of scalar floating point geps should be one. We cannot fold the address
  ; computation.
;CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds float*
  %a4 = getelementptr inbounds float* undef, i32 0
;CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds double*
  %a5 = getelementptr inbounds double* undef, i32 0


  ; Cost of vector geps should be one. We cannot fold the address computation.
;CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds <4 x i8>*
  %a7 = getelementptr inbounds <4 x i8>* undef, i32 0
;CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds <4 x i16>*
  %a8 = getelementptr inbounds <4 x i16>* undef, i32 0
;CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds <4 x i32>*
  %a9 = getelementptr inbounds <4 x i32>* undef, i32 0
;CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds <4 x i64>*
  %a10 = getelementptr inbounds <4 x i64>* undef, i32 0
;CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds <4 x float>*
  %a11 = getelementptr inbounds <4 x float>* undef, i32 0
;CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds <4 x double>*
  %a12 = getelementptr inbounds <4 x double>* undef, i32 0


  ret void
}
