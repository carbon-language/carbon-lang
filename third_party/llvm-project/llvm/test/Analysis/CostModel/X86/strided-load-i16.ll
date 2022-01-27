; REQUIRES: asserts
; RUN: opt -loop-vectorize -S -mattr=avx512bw --debug-only=loop-vectorize < %s 2>&1| FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@A = global [10240 x i16] zeroinitializer, align 16
@B = global [10240 x i16] zeroinitializer, align 16

; Function Attrs: nounwind uwtable
define void @load_i16_stride2() {
;CHECK-LABEL: load_i16_stride2
;CHECK: Found an estimated cost of 1 for VF 1 For instruction:   %1 = load
;CHECK: Found an estimated cost of 1 for VF 2 For instruction:   %1 = load
;CHECK: Found an estimated cost of 1 for VF 4 For instruction:   %1 = load
;CHECK: Found an estimated cost of 2 for VF 8 For instruction:   %1 = load
;CHECK: Found an estimated cost of 2 for VF 16 For instruction:   %1 = load
;CHECK: Found an estimated cost of 3 for VF 32 For instruction:   %1 = load
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %0 = shl nsw i64 %indvars.iv, 1
  %arrayidx = getelementptr inbounds [10240 x i16], [10240 x i16]* @A, i64 0, i64 %0
  %1 = load i16, i16* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds [10240 x i16], [10240 x i16]* @B, i64 0, i64 %indvars.iv
  store i16 %1, i16* %arrayidx2, align 2
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

define void @load_i16_stride3() {
;CHECK-LABEL: load_i16_stride3
;CHECK: Found an estimated cost of 1 for VF 1 For instruction:   %1 = load
;CHECK: Found an estimated cost of 1 for VF 2 For instruction:   %1 = load
;CHECK: Found an estimated cost of 2 for VF 4 For instruction:   %1 = load
;CHECK: Found an estimated cost of 2 for VF 8 For instruction:   %1 = load
;CHECK: Found an estimated cost of 3 for VF 16 For instruction:   %1 = load
;CHECK: Found an estimated cost of 5 for VF 32 For instruction:   %1 = load
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %0 = mul nsw i64 %indvars.iv, 3
  %arrayidx = getelementptr inbounds [10240 x i16], [10240 x i16]* @A, i64 0, i64 %0
  %1 = load i16, i16* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds [10240 x i16], [10240 x i16]* @B, i64 0, i64 %indvars.iv
  store i16 %1, i16* %arrayidx2, align 2
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

define void @load_i16_stride4() {
;CHECK-LABEL: load_i16_stride4
;CHECK: Found an estimated cost of 1 for VF 1 For instruction:   %1 = load
;CHECK: Found an estimated cost of 1 for VF 2 For instruction:   %1 = load
;CHECK: Found an estimated cost of 2 for VF 4 For instruction:   %1 = load
;CHECK: Found an estimated cost of 2 for VF 8 For instruction:   %1 = load
;CHECK: Found an estimated cost of 3 for VF 16 For instruction:   %1 = load
;CHECK: Found an estimated cost of 8 for VF 32 For instruction:   %1 = load
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %0 = shl nsw i64 %indvars.iv, 2
  %arrayidx = getelementptr inbounds [10240 x i16], [10240 x i16]* @A, i64 0, i64 %0
  %1 = load i16, i16* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds [10240 x i16], [10240 x i16]* @B, i64 0, i64 %indvars.iv
  store i16 %1, i16* %arrayidx2, align 2
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

define void @load_i16_stride5() {
;CHECK-LABEL: load_i16_stride5
;CHECK: Found an estimated cost of 1 for VF 1 For instruction:   %1 = load
;CHECK: Found an estimated cost of 2 for VF 2 For instruction:   %1 = load
;CHECK: Found an estimated cost of 2 for VF 4 For instruction:   %1 = load
;CHECK: Found an estimated cost of 3 for VF 8 For instruction:   %1 = load
;CHECK: Found an estimated cost of 5 for VF 16 For instruction:   %1 = load
;CHECK: Found an estimated cost of 10 for VF 32 For instruction:   %1 = load
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %0 = mul nsw i64 %indvars.iv, 5
  %arrayidx = getelementptr inbounds [10240 x i16], [10240 x i16]* @A, i64 0, i64 %0
  %1 = load i16, i16* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds [10240 x i16], [10240 x i16]* @B, i64 0, i64 %indvars.iv
  store i16 %1, i16* %arrayidx2, align 2
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}
