; RUN: opt < %s -basicaa -slp-vectorizer -dce -S -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"


; int foo(int *A) {
;   int r = A[0], g = A[1], b = A[2];
;   for (int i=0; i < A[13]; i++)
;     r*=18; g*=19; b*=12;
;   A[0] = r; A[1] = g; A[2] = b;
; }

;CHECK-LABEL: @foo
;CHECK: bitcast i32* %A to <3 x i32>*
;CHECK-NEXT: load <3 x i32>
;CHECK: phi <3 x i32>
;CHECK-NEXT: mul <3 x i32>
;CHECK-NOT: mul
;CHECK: phi <3 x i32>
;CHECK: bitcast i32* %A to <3 x i32>*
;CHECK-NEXT: store <3 x i32>
;CHECK-NEXT:ret i32 undef
define i32 @foo(i32* nocapture %A) {
entry:
  %0 = load i32* %A, align 4
  %arrayidx1 = getelementptr inbounds i32* %A, i64 1
  %1 = load i32* %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds i32* %A, i64 2
  %2 = load i32* %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds i32* %A, i64 13
  %3 = load i32* %arrayidx3, align 4
  %cmp18 = icmp sgt i32 %3, 0
  br i1 %cmp18, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %i.022 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %b.021 = phi i32 [ %mul5, %for.body ], [ %2, %entry ]
  %g.020 = phi i32 [ %mul4, %for.body ], [ %1, %entry ]
  %r.019 = phi i32 [ %mul, %for.body ], [ %0, %entry ]
  %mul = mul nsw i32 %r.019, 18
  %mul4 = mul nsw i32 %g.020, 19
  %mul5 = mul nsw i32 %b.021, 12
  %inc = add nsw i32 %i.022, 1
  %cmp = icmp slt i32 %inc, %3
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  %b.0.lcssa = phi i32 [ %2, %entry ], [ %mul5, %for.body ]
  %g.0.lcssa = phi i32 [ %1, %entry ], [ %mul4, %for.body ]
  %r.0.lcssa = phi i32 [ %0, %entry ], [ %mul, %for.body ]
  store i32 %r.0.lcssa, i32* %A, align 4
  store i32 %g.0.lcssa, i32* %arrayidx1, align 4
  store i32 %b.0.lcssa, i32* %arrayidx2, align 4
  ret i32 undef
}


