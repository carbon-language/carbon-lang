; RUN: opt -S -debug-only=loop-vectorize -loop-vectorize -instcombine  < %s 2>&1 | FileCheck %s
; REQUIRES: asserts

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "armv8--linux-gnueabihf"

@AB = common global [1024 x i8] zeroinitializer, align 4
@CD = common global [1024 x i8] zeroinitializer, align 4

define void @test_byte_interleaved_cost(i8 %C, i8 %D) {
entry:
  br label %for.body

; 8xi8 and 16xi8 are valid i8 vector types, so the cost of the interleaved
; access group is 2.

; CHECK: LV: Found an estimated cost of 2 for VF 8 For instruction:   %tmp = load i8, i8* %arrayidx0, align 4
; CHECK: LV: Found an estimated cost of 2 for VF 16 For instruction:   %tmp = load i8, i8* %arrayidx0, align 4

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx0 = getelementptr inbounds [1024 x i8], [1024 x i8]* @AB, i64 0, i64 %indvars.iv
  %tmp = load i8, i8* %arrayidx0, align 4
  %tmp1 = or i64 %indvars.iv, 1
  %arrayidx1 = getelementptr inbounds [1024 x i8], [1024 x i8]* @AB, i64 0, i64 %tmp1
  %tmp2 = load i8, i8* %arrayidx1, align 4
  %add = add nsw i8 %tmp, %C
  %mul = mul nsw i8 %tmp2, %D
  %arrayidx2 = getelementptr inbounds [1024 x i8], [1024 x i8]* @CD, i64 0, i64 %indvars.iv
  store i8 %add, i8* %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds [1024 x i8], [1024 x i8]* @CD, i64 0, i64 %tmp1
  store i8 %mul, i8* %arrayidx3, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 2
  %cmp = icmp slt i64 %indvars.iv.next, 1024
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}
