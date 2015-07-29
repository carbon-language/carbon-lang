; Check that we don't crash on corner cases.
; RUN: opt < %s -S -loop-unroll -unroll-max-iteration-count-to-analyze=1000 -unroll-threshold=10 -unroll-percent-dynamic-cost-saved-threshold=20 -o /dev/null
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

@known_constant = internal unnamed_addr constant [10 x i32] [i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1], align 16

define void @foo1() {
entry:
  br label %for.body

for.body:
  %phi = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %idx = zext i32 undef to i64
  %add.ptr = getelementptr inbounds i64, i64* null, i64 %idx
  %inc = add nuw nsw i64 %phi, 1
  %cmp = icmp ult i64 %inc, 999
  br i1 %cmp, label %for.body, label %for.exit

for.exit:
  ret void
}

define void @foo2() {
entry:
  br label %for.body

for.body:
  %phi = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %x = getelementptr i32, <4 x i32*> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %inc = add nuw nsw i64 %phi, 1
  %cmp = icmp ult i64 %inc, 999
  br i1 %cmp, label %for.body, label %for.exit

for.exit:
  ret void
}

define void @cmp_undef() {
entry:
  br label %for.body

for.body:                                         ; preds = %for.inc, %entry
  %iv.0 = phi i64 [ 0, %entry ], [ %iv.1, %for.inc ]
  %arrayidx1 = getelementptr inbounds [10 x i32], [10 x i32]* @known_constant, i64 0, i64 %iv.0
  %x1 = load i32, i32* %arrayidx1, align 4
  %cmp = icmp eq i32 %x1, undef
  br i1 %cmp, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %iv.1 = add nuw nsw i64 %iv.0, 1
  %exitcond = icmp eq i64 %iv.1, 10
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.inc
  ret void
}
