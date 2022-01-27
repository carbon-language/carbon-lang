; RUN: opt -loop-vectorize -vectorizer-maximize-bandwidth -S -mattr=+sse2 --debug-only=loop-vectorize < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,SSE2
; RUN: opt -loop-vectorize -vectorizer-maximize-bandwidth -S -mattr=+avx  --debug-only=loop-vectorize < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,AVX1
; RUN: opt -loop-vectorize -vectorizer-maximize-bandwidth -S -mattr=+avx2 --debug-only=loop-vectorize < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,AVX2
; RUN: opt -loop-vectorize -vectorizer-maximize-bandwidth -S -mattr=+avx512bw,+avx512vl --debug-only=loop-vectorize < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,AVX512
; REQUIRES: asserts

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@A = global [1024 x i8] zeroinitializer, align 128
@B = global [1024 x i8] zeroinitializer, align 128

; CHECK: LV: Checking a loop in "test"
;
; SSE2: LV: Found an estimated cost of 1 for VF 1 For instruction:   store i8 %v2, i8* %out2, align 1
; SSE2: LV: Found an estimated cost of 25 for VF 2 For instruction:   store i8 %v2, i8* %out2, align 1
; SSE2: LV: Found an estimated cost of 52 for VF 4 For instruction:   store i8 %v2, i8* %out2, align 1
; SSE2: LV: Found an estimated cost of 101 for VF 8 For instruction:   store i8 %v2, i8* %out2, align 1
; SSE2: LV: Found an estimated cost of 204 for VF 16 For instruction:   store i8 %v2, i8* %out2, align 1
;
; AVX1: LV: Found an estimated cost of 1 for VF 1 For instruction:   store i8 %v2, i8* %out2, align 1
; AVX1: LV: Found an estimated cost of 15 for VF 2 For instruction:   store i8 %v2, i8* %out2, align 1
; AVX1: LV: Found an estimated cost of 27 for VF 4 For instruction:   store i8 %v2, i8* %out2, align 1
; AVX1: LV: Found an estimated cost of 54 for VF 8 For instruction:   store i8 %v2, i8* %out2, align 1
; AVX1: LV: Found an estimated cost of 101 for VF 16 For instruction:   store i8 %v2, i8* %out2, align 1
; AVX1: LV: Found an estimated cost of 249 for VF 32 For instruction:   store i8 %v2, i8* %out2, align 1
;
; AVX2: LV: Found an estimated cost of 1 for VF 1 For instruction:   store i8 %v2, i8* %out2, align 1
; AVX2: LV: Found an estimated cost of 7 for VF 2 For instruction:   store i8 %v2, i8* %out2, align 1
; AVX2: LV: Found an estimated cost of 7 for VF 4 For instruction:   store i8 %v2, i8* %out2, align 1
; AVX2: LV: Found an estimated cost of 9 for VF 8 For instruction:   store i8 %v2, i8* %out2, align 1
; AVX2: LV: Found an estimated cost of 13 for VF 16 For instruction:   store i8 %v2, i8* %out2, align 1
; AVX2: LV: Found an estimated cost of 16 for VF 32 For instruction:   store i8 %v2, i8* %out2, align 1
;
; AVX512: LV: Found an estimated cost of 1 for VF 1 For instruction:   store i8 %v2, i8* %out2, align 1
; AVX512: LV: Found an estimated cost of 8 for VF 2 For instruction:   store i8 %v2, i8* %out2, align 1
; AVX512: LV: Found an estimated cost of 8 for VF 4 For instruction:   store i8 %v2, i8* %out2, align 1
; AVX512: LV: Found an estimated cost of 16 for VF 8 For instruction:   store i8 %v2, i8* %out2, align 1
; AVX512: LV: Found an estimated cost of 13 for VF 16 For instruction:   store i8 %v2, i8* %out2, align 1
; AVX512: LV: Found an estimated cost of 16 for VF 32 For instruction:   store i8 %v2, i8* %out2, align 1
; AVX512: LV: Found an estimated cost of 29 for VF 64 For instruction:   store i8 %v2, i8* %out2, align 1
;
; CHECK-NOT: LV: Found an estimated cost of {{[0-9]+}} for VF {{[0-9]+}} For instruction:   store i8 %v2, i8* %out2, align 1

define void @test() {
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]

  %iv.0 = add nuw nsw i64 %iv, 0
  %iv.1 = add nuw nsw i64 %iv, 1
  %iv.2 = add nuw nsw i64 %iv, 2

  %in = getelementptr inbounds [1024 x i8], [1024 x i8]* @A, i64 0, i64 %iv.0
  %v = load i8, i8* %in

  %v0 = add i8 %v, 0
  %v1 = add i8 %v, 1
  %v2 = add i8 %v, 2

  %out0 = getelementptr inbounds [1024 x i8], [1024 x i8]* @B, i64 0, i64 %iv.0
  %out1 = getelementptr inbounds [1024 x i8], [1024 x i8]* @B, i64 0, i64 %iv.1
  %out2 = getelementptr inbounds [1024 x i8], [1024 x i8]* @B, i64 0, i64 %iv.2

  store i8 %v0, i8* %out0
  store i8 %v1, i8* %out1
  store i8 %v2, i8* %out2

  %iv.next = add nuw nsw i64 %iv.0, 3
  %cmp = icmp ult i64 %iv.next, 1024
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void
}
