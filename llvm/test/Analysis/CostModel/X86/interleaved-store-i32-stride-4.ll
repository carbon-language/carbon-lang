; RUN: opt -loop-vectorize -vectorizer-maximize-bandwidth -S -mattr=+sse2 --debug-only=loop-vectorize < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,SSE2
; RUN: opt -loop-vectorize -vectorizer-maximize-bandwidth -S -mattr=+avx  --debug-only=loop-vectorize < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,AVX1
; RUN: opt -loop-vectorize -vectorizer-maximize-bandwidth -S -mattr=+avx2 --debug-only=loop-vectorize < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,AVX2
; RUN: opt -loop-vectorize -vectorizer-maximize-bandwidth -S -mattr=+avx512bw,+avx512vl --debug-only=loop-vectorize < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,AVX512
; REQUIRES: asserts

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@A = global [1024 x i8] zeroinitializer, align 128
@B = global [1024 x i32] zeroinitializer, align 128

; CHECK: LV: Checking a loop in "test"
;
; SSE2: LV: Found an estimated cost of 1 for VF 1 For instruction:   store i32 %v3, i32* %out3, align 4
; SSE2: LV: Found an estimated cost of 28 for VF 2 For instruction:   store i32 %v3, i32* %out3, align 4
; SSE2: LV: Found an estimated cost of 60 for VF 4 For instruction:   store i32 %v3, i32* %out3, align 4
; SSE2: LV: Found an estimated cost of 120 for VF 8 For instruction:   store i32 %v3, i32* %out3, align 4
; SSE2: LV: Found an estimated cost of 240 for VF 16 For instruction:   store i32 %v3, i32* %out3, align 4
;
; AVX1: LV: Found an estimated cost of 1 for VF 1 For instruction:   store i32 %v3, i32* %out3, align 4
; AVX1: LV: Found an estimated cost of 19 for VF 2 For instruction:   store i32 %v3, i32* %out3, align 4
; AVX1: LV: Found an estimated cost of 38 for VF 4 For instruction:   store i32 %v3, i32* %out3, align 4
; AVX1: LV: Found an estimated cost of 92 for VF 8 For instruction:   store i32 %v3, i32* %out3, align 4
; AVX1: LV: Found an estimated cost of 184 for VF 16 For instruction:   store i32 %v3, i32* %out3, align 4
; AVX1: LV: Found an estimated cost of 368 for VF 32 For instruction:   store i32 %v3, i32* %out3, align 4
;
; AVX2: LV: Found an estimated cost of 1 for VF 1 For instruction:   store i32 %v3, i32* %out3, align 4
; AVX2: LV: Found an estimated cost of 6 for VF 2 For instruction:   store i32 %v3, i32* %out3, align 4
; AVX2: LV: Found an estimated cost of 8 for VF 4 For instruction:   store i32 %v3, i32* %out3, align 4
; AVX2: LV: Found an estimated cost of 20 for VF 8 For instruction:   store i32 %v3, i32* %out3, align 4
; AVX2: LV: Found an estimated cost of 40 for VF 16 For instruction:   store i32 %v3, i32* %out3, align 4
; AVX2: LV: Found an estimated cost of 80 for VF 32 For instruction:   store i32 %v3, i32* %out3, align 4
;
; AVX512: LV: Found an estimated cost of 1 for VF 1 For instruction:   store i32 %v3, i32* %out3, align 4
; AVX512: LV: Found an estimated cost of 5 for VF 2 For instruction:   store i32 %v3, i32* %out3, align 4
; AVX512: LV: Found an estimated cost of 5 for VF 4 For instruction:   store i32 %v3, i32* %out3, align 4
; AVX512: LV: Found an estimated cost of 11 for VF 8 For instruction:   store i32 %v3, i32* %out3, align 4
; AVX512: LV: Found an estimated cost of 22 for VF 16 For instruction:   store i32 %v3, i32* %out3, align 4
; AVX512: LV: Found an estimated cost of 44 for VF 32 For instruction:   store i32 %v3, i32* %out3, align 4
; AVX512: LV: Found an estimated cost of 88 for VF 64 For instruction:   store i32 %v3, i32* %out3, align 4
;
; CHECK-NOT: LV: Found an estimated cost of {{[0-9]+}} for VF {{[0-9]+}} For instruction:   store i32 %v3, i32* %out3, align 4

define void @test() {
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]

  %iv.0 = add nuw nsw i64 %iv, 0
  %iv.1 = add nuw nsw i64 %iv, 1
  %iv.2 = add nuw nsw i64 %iv, 2
  %iv.3 = add nuw nsw i64 %iv, 3

  %in = getelementptr inbounds [1024 x i8], [1024 x i8]* @A, i64 0, i64 %iv.0
  %v.narrow = load i8, i8* %in

  %v = zext i8 %v.narrow to i32

  %v0 = add i32 %v, 0
  %v1 = add i32 %v, 1
  %v2 = add i32 %v, 2
  %v3 = add i32 %v, 3

  %out0 = getelementptr inbounds [1024 x i32], [1024 x i32]* @B, i64 0, i64 %iv.0
  %out1 = getelementptr inbounds [1024 x i32], [1024 x i32]* @B, i64 0, i64 %iv.1
  %out2 = getelementptr inbounds [1024 x i32], [1024 x i32]* @B, i64 0, i64 %iv.2
  %out3 = getelementptr inbounds [1024 x i32], [1024 x i32]* @B, i64 0, i64 %iv.3

  store i32 %v0, i32* %out0
  store i32 %v1, i32* %out1
  store i32 %v2, i32* %out2
  store i32 %v3, i32* %out3

  %iv.next = add nuw nsw i64 %iv.0, 4
  %cmp = icmp ult i64 %iv.next, 1024
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void
}
