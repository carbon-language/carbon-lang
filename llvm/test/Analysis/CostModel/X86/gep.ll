; RUN: opt < %s  -cost-model -analyze -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"


define void @test_geps() {
  ; Cost of should be zero. We expect it to be folded into
  ; the instruction addressing mode.
;CHECK:  cost of 0 for instruction: {{.*}} getelementptr inbounds i8*
  %a0 = getelementptr inbounds i8* undef, i32 0
;CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i16*
  %a1 = getelementptr inbounds i16* undef, i32 0
;CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i32*
  %a2 = getelementptr inbounds i32* undef, i32 0
;CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i64*
  %a3 = getelementptr inbounds i64* undef, i32 0

;CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds float*
  %a4 = getelementptr inbounds float* undef, i32 0
;CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds double*
  %a5 = getelementptr inbounds double* undef, i32 0

 ; Vector geps should also have zero cost.
;CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds <4 x i8>*
  %a7 = getelementptr inbounds <4 x i8>* undef, i32 0
;CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds <4 x i16>*
  %a8 = getelementptr inbounds <4 x i16>* undef, i32 0
;CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds <4 x i32>*
  %a9 = getelementptr inbounds <4 x i32>* undef, i32 0
;CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds <4 x i64>*
  %a10 = getelementptr inbounds <4 x i64>* undef, i32 0
;CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds <4 x float>*
  %a11 = getelementptr inbounds <4 x float>* undef, i32 0
;CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds <4 x double>*
  %a12 = getelementptr inbounds <4 x double>* undef, i32 0


  ret void
}
