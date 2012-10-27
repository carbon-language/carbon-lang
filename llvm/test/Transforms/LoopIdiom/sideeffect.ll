; RUN: opt -basicaa -loop-idiom < %s -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

; PR9481
define i32 @test1() nounwind uwtable ssp {
entry:
  %a = alloca [10 x i8], align 1
  br label %for.body

for.cond1.preheader:                              ; preds = %for.body
  %arrayidx5.phi.trans.insert = getelementptr inbounds [10 x i8]* %a, i64 0, i64 0
  %.pre = load i8* %arrayidx5.phi.trans.insert, align 1
  br label %for.body3

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv29 = phi i64 [ 0, %entry ], [ %indvars.iv.next30, %for.body ]
  call void (...)* @bar() nounwind
  %arrayidx = getelementptr inbounds [10 x i8]* %a, i64 0, i64 %indvars.iv29
  store i8 23, i8* %arrayidx, align 1
  %indvars.iv.next30 = add i64 %indvars.iv29, 1
  %lftr.wideiv31 = trunc i64 %indvars.iv.next30 to i32
  %exitcond32 = icmp eq i32 %lftr.wideiv31, 1000000
  br i1 %exitcond32, label %for.cond1.preheader, label %for.body

for.body3:                                        ; preds = %for.body3, %for.cond1.preheader
  %0 = phi i8 [ %.pre, %for.cond1.preheader ], [ %add, %for.body3 ]
  %indvars.iv = phi i64 [ 1, %for.cond1.preheader ], [ %indvars.iv.next, %for.body3 ]
  call void (...)* @bar() nounwind
  %arrayidx7 = getelementptr inbounds [10 x i8]* %a, i64 0, i64 %indvars.iv
  %1 = load i8* %arrayidx7, align 1
  %add = add i8 %1, %0
  store i8 %add, i8* %arrayidx7, align 1
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1000000
  br i1 %exitcond, label %for.end12, label %for.body3

for.end12:                                        ; preds = %for.body3
  %arrayidx13 = getelementptr inbounds [10 x i8]* %a, i64 0, i64 2
  %2 = load i8* %arrayidx13, align 1
  %conv14 = sext i8 %2 to i32
  %arrayidx15 = getelementptr inbounds [10 x i8]* %a, i64 0, i64 6
  %3 = load i8* %arrayidx15, align 1
  %conv16 = sext i8 %3 to i32
  %add17 = add nsw i32 %conv16, %conv14
  ret i32 %add17

; CHECK: @test1
; CHECK-NOT: @llvm.memset
}

declare void @bar(...)
