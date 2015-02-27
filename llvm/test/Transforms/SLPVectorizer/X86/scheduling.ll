; RUN: opt < %s -basicaa -slp-vectorizer -S -mtriple=i386-apple-macosx10.8.0 -mcpu=corei7-avx | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

;CHECK-LABEL: @foo
;CHECK: load <4 x i32>
;CHECK: load <4 x i32>
;CHECK: %[[S1:.+]] = add nsw <4 x i32>
;CHECK-DAG: store <4 x i32> %[[S1]]
;CHECK-DAG: %[[A1:.+]] = add nsw i32
;CHECK-DAG: %[[A2:.+]] = add nsw i32 %[[A1]]
;CHECK-DAG: %[[A3:.+]] = add nsw i32 %[[A2]]
;CHECK-DAG: %[[A4:.+]] = add nsw i32 %[[A3]]
;CHECK: ret i32 %[[A4]] 

define i32 @foo(i32* nocapture readonly %diff) #0 {
entry:
  %m2 = alloca [8 x [8 x i32]], align 16
  %0 = bitcast [8 x [8 x i32]]* %m2 to i8*
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %a.088 = phi i32 [ 0, %entry ], [ %add52, %for.body ]
  %1 = shl i64 %indvars.iv, 3
  %arrayidx = getelementptr inbounds i32, i32* %diff, i64 %1
  %2 = load i32, i32* %arrayidx, align 4
  %3 = or i64 %1, 4
  %arrayidx2 = getelementptr inbounds i32, i32* %diff, i64 %3
  %4 = load i32, i32* %arrayidx2, align 4
  %add3 = add nsw i32 %4, %2
  %arrayidx6 = getelementptr inbounds [8 x [8 x i32]], [8 x [8 x i32]]* %m2, i64 0, i64 %indvars.iv, i64 0
  store i32 %add3, i32* %arrayidx6, align 16
  %add10 = add nsw i32 %add3, %a.088
  %5 = or i64 %1, 1
  %arrayidx13 = getelementptr inbounds i32, i32* %diff, i64 %5
  %6 = load i32, i32* %arrayidx13, align 4
  %7 = or i64 %1, 5
  %arrayidx16 = getelementptr inbounds i32, i32* %diff, i64 %7
  %8 = load i32, i32* %arrayidx16, align 4
  %add17 = add nsw i32 %8, %6
  %arrayidx20 = getelementptr inbounds [8 x [8 x i32]], [8 x [8 x i32]]* %m2, i64 0, i64 %indvars.iv, i64 1
  store i32 %add17, i32* %arrayidx20, align 4
  %add24 = add nsw i32 %add10, %add17
  %9 = or i64 %1, 2
  %arrayidx27 = getelementptr inbounds i32, i32* %diff, i64 %9
  %10 = load i32, i32* %arrayidx27, align 4
  %11 = or i64 %1, 6
  %arrayidx30 = getelementptr inbounds i32, i32* %diff, i64 %11
  %12 = load i32, i32* %arrayidx30, align 4
  %add31 = add nsw i32 %12, %10
  %arrayidx34 = getelementptr inbounds [8 x [8 x i32]], [8 x [8 x i32]]* %m2, i64 0, i64 %indvars.iv, i64 2
  store i32 %add31, i32* %arrayidx34, align 8
  %add38 = add nsw i32 %add24, %add31
  %13 = or i64 %1, 3
  %arrayidx41 = getelementptr inbounds i32, i32* %diff, i64 %13
  %14 = load i32, i32* %arrayidx41, align 4
  %15 = or i64 %1, 7
  %arrayidx44 = getelementptr inbounds i32, i32* %diff, i64 %15
  %16 = load i32, i32* %arrayidx44, align 4
  %add45 = add nsw i32 %16, %14
  %arrayidx48 = getelementptr inbounds [8 x [8 x i32]], [8 x [8 x i32]]* %m2, i64 0, i64 %indvars.iv, i64 3
  store i32 %add45, i32* %arrayidx48, align 4
  %add52 = add nsw i32 %add38, %add45
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 8
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  %arraydecay = getelementptr inbounds [8 x [8 x i32]], [8 x [8 x i32]]* %m2, i64 0, i64 0
  call void @ff([8 x i32]* %arraydecay) #1
  ret i32 %add52
}

declare void @ff([8 x i32]*) #2


