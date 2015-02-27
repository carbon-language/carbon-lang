; RUN: opt < %s -basicaa -slp-vectorizer -dce -S -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

; int foo(int *A) {
;   int r = A[0], g = A[1], b = A[2], a = A[3];
;   for (int i=0; i < A[13]; i++) {
;     r*=18; g*=19; b*=12; a *=9;
;   }
;   A[0] = r; A[1] = g; A[2] = b; A[3] = a;
; }

;CHECK-LABEL: @foo
;CHECK: bitcast i32* %A to <4 x i32>*
;CHECK-NEXT: load <4 x i32>
;CHECK: phi <4 x i32>
;CHECK-NEXT: mul nsw <4 x i32>
;CHECK-NOT: mul
;CHECK: phi <4 x i32>
;CHECK: bitcast i32* %A to <4 x i32>*
;CHECK-NEXT: store <4 x i32>
;CHECK-NEXT:ret i32 undef
define i32 @foo(i32* nocapture %A) #0 {
entry:
  %0 = load i32* %A, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %A, i64 1
  %1 = load i32* %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i64 2
  %2 = load i32* %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %A, i64 3
  %3 = load i32* %arrayidx3, align 4
  %arrayidx4 = getelementptr inbounds i32, i32* %A, i64 13
  %4 = load i32* %arrayidx4, align 4
  %cmp24 = icmp sgt i32 %4, 0
  br i1 %cmp24, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %i.029 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %a.028 = phi i32 [ %mul7, %for.body ], [ %3, %entry ]
  %b.027 = phi i32 [ %mul6, %for.body ], [ %2, %entry ]
  %g.026 = phi i32 [ %mul5, %for.body ], [ %1, %entry ]
  %r.025 = phi i32 [ %mul, %for.body ], [ %0, %entry ]
  %mul = mul nsw i32 %r.025, 18
  %mul5 = mul nsw i32 %g.026, 19
  %mul6 = mul nsw i32 %b.027, 12
  %mul7 = mul nsw i32 %a.028, 9
  %inc = add nsw i32 %i.029, 1
  %cmp = icmp slt i32 %inc, %4
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  %a.0.lcssa = phi i32 [ %3, %entry ], [ %mul7, %for.body ]
  %b.0.lcssa = phi i32 [ %2, %entry ], [ %mul6, %for.body ]
  %g.0.lcssa = phi i32 [ %1, %entry ], [ %mul5, %for.body ]
  %r.0.lcssa = phi i32 [ %0, %entry ], [ %mul, %for.body ]
  store i32 %r.0.lcssa, i32* %A, align 4
  store i32 %g.0.lcssa, i32* %arrayidx1, align 4
  store i32 %b.0.lcssa, i32* %arrayidx2, align 4
  store i32 %a.0.lcssa, i32* %arrayidx3, align 4
  ret i32 undef
}


