; RUN: opt -loop-vectorize -vectorizer-maximize-bandwidth -S -mattr=+sse2 --debug-only=loop-vectorize < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,SSE2
; RUN: opt -loop-vectorize -vectorizer-maximize-bandwidth -S -mattr=+avx  --debug-only=loop-vectorize < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,AVX1
; RUN: opt -loop-vectorize -vectorizer-maximize-bandwidth -S -mattr=+avx2 --debug-only=loop-vectorize < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,AVX2
; RUN: opt -loop-vectorize -vectorizer-maximize-bandwidth -S -mattr=+avx512bw,+avx512vl --debug-only=loop-vectorize < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,AVX512
; REQUIRES: asserts

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@A = global [1024 x i8] zeroinitializer, align 128
@B = global [1024 x float] zeroinitializer, align 128

; CHECK: LV: Checking a loop in "test"
;
; SSE2: LV: Found an estimated cost of 1 for VF 1 For instruction:   store float %v2, float* %out2, align 4
; SSE2: LV: Found an estimated cost of 11 for VF 2 For instruction:   store float %v2, float* %out2, align 4
; SSE2: LV: Found an estimated cost of 24 for VF 4 For instruction:   store float %v2, float* %out2, align 4
; SSE2: LV: Found an estimated cost of 48 for VF 8 For instruction:   store float %v2, float* %out2, align 4
; SSE2: LV: Found an estimated cost of 96 for VF 16 For instruction:   store float %v2, float* %out2, align 4
;
; AVX1: LV: Found an estimated cost of 1 for VF 1 For instruction:   store float %v2, float* %out2, align 4
; AVX1: LV: Found an estimated cost of 13 for VF 2 For instruction:   store float %v2, float* %out2, align 4
; AVX1: LV: Found an estimated cost of 23 for VF 4 For instruction:   store float %v2, float* %out2, align 4
; AVX1: LV: Found an estimated cost of 57 for VF 8 For instruction:   store float %v2, float* %out2, align 4
; AVX1: LV: Found an estimated cost of 114 for VF 16 For instruction:   store float %v2, float* %out2, align 4
; AVX1: LV: Found an estimated cost of 228 for VF 32 For instruction:   store float %v2, float* %out2, align 4
;
; AVX2: LV: Found an estimated cost of 1 for VF 1 For instruction:   store float %v2, float* %out2, align 4
; AVX2: LV: Found an estimated cost of 7 for VF 2 For instruction:   store float %v2, float* %out2, align 4
; AVX2: LV: Found an estimated cost of 7 for VF 4 For instruction:   store float %v2, float* %out2, align 4
; AVX2: LV: Found an estimated cost of 14 for VF 8 For instruction:   store float %v2, float* %out2, align 4
; AVX2: LV: Found an estimated cost of 28 for VF 16 For instruction:   store float %v2, float* %out2, align 4
; AVX2: LV: Found an estimated cost of 60 for VF 32 For instruction:   store float %v2, float* %out2, align 4
;
; AVX512: LV: Found an estimated cost of 1 for VF 1 For instruction:   store float %v2, float* %out2, align 4
; AVX512: LV: Found an estimated cost of 4 for VF 2 For instruction:   store float %v2, float* %out2, align 4
; AVX512: LV: Found an estimated cost of 4 for VF 4 For instruction:   store float %v2, float* %out2, align 4
; AVX512: LV: Found an estimated cost of 8 for VF 8 For instruction:   store float %v2, float* %out2, align 4
; AVX512: LV: Found an estimated cost of 12 for VF 16 For instruction:   store float %v2, float* %out2, align 4
; AVX512: LV: Found an estimated cost of 24 for VF 32 For instruction:   store float %v2, float* %out2, align 4
; AVX512: LV: Found an estimated cost of 48 for VF 64 For instruction:   store float %v2, float* %out2, align 4
;
; CHECK-NOT: LV: Found an estimated cost of {{[0-9]+}} for VF {{[0-9]+}} For instruction:   store float %v2, float* %out2, align 4

define void @test() {
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]

  %iv.0 = add nuw nsw i64 %iv, 0
  %iv.1 = add nuw nsw i64 %iv, 1
  %iv.2 = add nuw nsw i64 %iv, 2

  %in = getelementptr inbounds [1024 x i8], [1024 x i8]* @A, i64 0, i64 %iv.0
  %v.narrow = load i8, i8* %in

  %v = uitofp i8 %v.narrow to float

  %v0 = fadd float %v, 0.0
  %v1 = fadd float %v, 1.0
  %v2 = fadd float %v, 2.0

  %out0 = getelementptr inbounds [1024 x float], [1024 x float]* @B, i64 0, i64 %iv.0
  %out1 = getelementptr inbounds [1024 x float], [1024 x float]* @B, i64 0, i64 %iv.1
  %out2 = getelementptr inbounds [1024 x float], [1024 x float]* @B, i64 0, i64 %iv.2

  store float %v0, float* %out0
  store float %v1, float* %out1
  store float %v2, float* %out2

  %iv.next = add nuw nsw i64 %iv.0, 3
  %cmp = icmp ult i64 %iv.next, 1024
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void
}
