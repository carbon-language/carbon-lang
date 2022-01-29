; RUN: opt < %s -loop-vectorize -dce -instcombine -pass-remarks=loop-vectorize -pass-remarks-analysis=loop-vectorize -pass-remarks-missed=loop-vectorize -S 2>&1 | FileCheck %s -check-prefix=OVERRIDE
; RUN: opt < %s -loop-vectorize -pragma-vectorize-memory-check-threshold=6 -dce -instcombine -pass-remarks=loop-vectorize -pass-remarks-analysis=loop-vectorize -pass-remarks-missed=loop-vectorize -S 2>&1 | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

target triple = "x86_64-unknown-linux"

; First loop produced diagnostic pass remark.
;CHECK: remark: {{.*}}:0:0: vectorized loop (vectorization width: 4, interleaved count: 2)
; Second loop produces diagnostic analysis remark.
;CHECK: remark: {{.*}}:0:0: loop not vectorized: cannot prove it is safe to reorder memory operations

; First loop produced diagnostic pass remark.
;OVERRIDE: remark: {{.*}}:0:0: vectorized loop (vectorization width: 4, interleaved count: 2)
; Second loop produces diagnostic pass remark.
;OVERRIDE: remark: {{.*}}:0:0: loop not vectorized: cannot prove it is safe to reorder memory operations

; We are vectorizing with 6 runtime checks.
;CHECK-LABEL: func1x6(
;CHECK: <4 x i32>
;CHECK: ret
;OVERRIDE-LABEL: func1x6(
;OVERRIDE: <4 x i32>
;OVERRIDE: ret
define i32 @func1x6(i32* nocapture %out, i32* nocapture %A, i32* nocapture %B, i32* nocapture %C, i32* nocapture %D, i32* nocapture %E, i32* nocapture %F) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.016 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %i.016
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %B, i64 %i.016
  %1 = load i32, i32* %arrayidx1, align 4
  %add = add nsw i32 %1, %0
  %arrayidx2 = getelementptr inbounds i32, i32* %C, i64 %i.016
  %2 = load i32, i32* %arrayidx2, align 4
  %add3 = add nsw i32 %add, %2
  %arrayidx4 = getelementptr inbounds i32, i32* %E, i64 %i.016
  %3 = load i32, i32* %arrayidx4, align 4
  %add5 = add nsw i32 %add3, %3
  %arrayidx6 = getelementptr inbounds i32, i32* %F, i64 %i.016
  %4 = load i32, i32* %arrayidx6, align 4
  %add7 = add nsw i32 %add5, %4
  %arrayidx8 = getelementptr inbounds i32, i32* %out, i64 %i.016
  store i32 %add7, i32* %arrayidx8, align 4
  %inc = add i64 %i.016, 1
  %exitcond = icmp eq i64 %inc, 256
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret i32 undef
}

; We are not vectorizing with 12 runtime checks.
;CHECK-LABEL: func2x6(
;CHECK-NOT: <4 x i32>
;CHECK: ret
; We vectorize with 12 checks if a vectorization hint is provided.
;OVERRIDE-LABEL: func2x6(
;OVERRIDE-NOT: <4 x i32>
;OVERRIDE: ret
define i32 @func2x6(i32* nocapture %out, i32* nocapture %out2, i32* nocapture %A, i32* nocapture %B, i32* nocapture %C, i32* nocapture %D, i32* nocapture %E, i32* nocapture %F) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.037 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %i.037
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %B, i64 %i.037
  %1 = load i32, i32* %arrayidx1, align 4
  %add = add nsw i32 %1, %0
  %arrayidx2 = getelementptr inbounds i32, i32* %C, i64 %i.037
  %2 = load i32, i32* %arrayidx2, align 4
  %add3 = add nsw i32 %add, %2
  %arrayidx4 = getelementptr inbounds i32, i32* %E, i64 %i.037
  %3 = load i32, i32* %arrayidx4, align 4
  %add5 = add nsw i32 %add3, %3
  %arrayidx6 = getelementptr inbounds i32, i32* %F, i64 %i.037
  %4 = load i32, i32* %arrayidx6, align 4
  %add7 = add nsw i32 %add5, %4
  %arrayidx8 = getelementptr inbounds i32, i32* %out, i64 %i.037
  store i32 %add7, i32* %arrayidx8, align 4
  %5 = load i32, i32* %arrayidx, align 4
  %6 = load i32, i32* %arrayidx1, align 4
  %add11 = add nsw i32 %6, %5
  %7 = load i32, i32* %arrayidx2, align 4
  %add13 = add nsw i32 %add11, %7
  %8 = load i32, i32* %arrayidx4, align 4
  %add15 = add nsw i32 %add13, %8
  %9 = load i32, i32* %arrayidx6, align 4
  %add17 = add nsw i32 %add15, %9
  %arrayidx18 = getelementptr inbounds i32, i32* %out2, i64 %i.037
  store i32 %add17, i32* %arrayidx18, align 4
  %inc = add i64 %i.037, 1
  %exitcond = icmp eq i64 %inc, 256
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret i32 undef
}

