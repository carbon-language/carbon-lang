; Check that we don't crash on corner cases.
; RUN: opt < %s -S -loop-unroll -unroll-max-iteration-count-to-analyze=1000 -unroll-absolute-threshold=10  -unroll-threshold=10  -unroll-percent-of-optimized-for-complete-unroll=20 -o /dev/null
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

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
