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
; SSE2: LV: Found an estimated cost of 1 for VF 1 For instruction:   %v0 = load i8, i8* %in0, align 1
; SSE2: LV: Found an estimated cost of 14 for VF 2 For instruction:   %v0 = load i8, i8* %in0, align 1
; SSE2: LV: Found an estimated cost of 30 for VF 4 For instruction:   %v0 = load i8, i8* %in0, align 1
; SSE2: LV: Found an estimated cost of 62 for VF 8 For instruction:   %v0 = load i8, i8* %in0, align 1
; SSE2: LV: Found an estimated cost of 126 for VF 16 For instruction:   %v0 = load i8, i8* %in0, align 1
;
; AVX1: LV: Found an estimated cost of 1 for VF 1 For instruction:   %v0 = load i8, i8* %in0, align 1
; AVX1: LV: Found an estimated cost of 9 for VF 2 For instruction:   %v0 = load i8, i8* %in0, align 1
; AVX1: LV: Found an estimated cost of 17 for VF 4 For instruction:   %v0 = load i8, i8* %in0, align 1
; AVX1: LV: Found an estimated cost of 33 for VF 8 For instruction:   %v0 = load i8, i8* %in0, align 1
; AVX1: LV: Found an estimated cost of 81 for VF 16 For instruction:   %v0 = load i8, i8* %in0, align 1
; AVX1: LV: Found an estimated cost of 166 for VF 32 For instruction:   %v0 = load i8, i8* %in0, align 1
;
; AVX2: LV: Found an estimated cost of 1 for VF 1 For instruction:   %v0 = load i8, i8* %in0, align 1
; AVX2: LV: Found an estimated cost of 3 for VF 2 For instruction:   %v0 = load i8, i8* %in0, align 1
; AVX2: LV: Found an estimated cost of 3 for VF 4 For instruction:   %v0 = load i8, i8* %in0, align 1
; AVX2: LV: Found an estimated cost of 3 for VF 8 For instruction:   %v0 = load i8, i8* %in0, align 1
; AVX2: LV: Found an estimated cost of 5 for VF 16 For instruction:   %v0 = load i8, i8* %in0, align 1
; AVX2: LV: Found an estimated cost of 8 for VF 32 For instruction:   %v0 = load i8, i8* %in0, align 1
;
; AVX512: LV: Found an estimated cost of 1 for VF 1 For instruction:   %v0 = load i8, i8* %in0, align 1
; AVX512: LV: Found an estimated cost of 3 for VF 2 For instruction:   %v0 = load i8, i8* %in0, align 1
; AVX512: LV: Found an estimated cost of 3 for VF 4 For instruction:   %v0 = load i8, i8* %in0, align 1
; AVX512: LV: Found an estimated cost of 3 for VF 8 For instruction:   %v0 = load i8, i8* %in0, align 1
; AVX512: LV: Found an estimated cost of 9 for VF 16 For instruction:   %v0 = load i8, i8* %in0, align 1
; AVX512: LV: Found an estimated cost of 17 for VF 32 For instruction:   %v0 = load i8, i8* %in0, align 1
; AVX512: LV: Found an estimated cost of 41 for VF 64 For instruction:   %v0 = load i8, i8* %in0, align 1
;
; CHECK-NOT: LV: Found an estimated cost of {{[0-9]+}} for VF {{[0-9]+}} For instruction:   %v0 = load i8, i8* %in0, align 1

define void @test() {
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]

  %iv.0 = add nuw nsw i64 %iv, 0
  %iv.1 = add nuw nsw i64 %iv, 1

  %in0 = getelementptr inbounds [1024 x i8], [1024 x i8]* @A, i64 0, i64 %iv.0
  %in1 = getelementptr inbounds [1024 x i8], [1024 x i8]* @A, i64 0, i64 %iv.1

  %v0 = load i8, i8* %in0
  %v1 = load i8, i8* %in1

  %reduce.add.0 = add i8 %v0, %v1

  %out = getelementptr inbounds [1024 x i8], [1024 x i8]* @B, i64 0, i64 %iv.0
  store i8 %reduce.add.0, i8* %out

  %iv.next = add nuw nsw i64 %iv.0, 2
  %cmp = icmp ult i64 %iv.next, 1024
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void
}
