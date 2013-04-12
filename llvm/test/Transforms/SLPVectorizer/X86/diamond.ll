; RUN: opt < %s -basicaa -slp-vectorizer -dce -S -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

; int foo(int * restrict B,  int * restrict A, int n, int m) {
;   B[0] = n * A[0] + m * A[0];
;   B[1] = n * A[1] + m * A[1];
;   B[2] = n * A[2] + m * A[2];
;   B[3] = n * A[3] + m * A[3];
;   return 0;
; }

; CHECK: @foo
; CHECK: load <4 x i32>
; CHECK: mul <4 x i32>
; CHECK: store <4 x i32>
; CHECK: ret
define i32 @foo(i32* noalias nocapture %B, i32* noalias nocapture %A, i32 %n, i32 %m) #0 {
entry:
  %0 = load i32* %A, align 4, !tbaa !0
  %mul238 = add i32 %m, %n
  %add = mul i32 %0, %mul238
  store i32 %add, i32* %B, align 4, !tbaa !0
  %arrayidx4 = getelementptr inbounds i32* %A, i64 1
  %1 = load i32* %arrayidx4, align 4, !tbaa !0
  %add8 = mul i32 %1, %mul238
  %arrayidx9 = getelementptr inbounds i32* %B, i64 1
  store i32 %add8, i32* %arrayidx9, align 4, !tbaa !0
  %arrayidx10 = getelementptr inbounds i32* %A, i64 2
  %2 = load i32* %arrayidx10, align 4, !tbaa !0
  %add14 = mul i32 %2, %mul238
  %arrayidx15 = getelementptr inbounds i32* %B, i64 2
  store i32 %add14, i32* %arrayidx15, align 4, !tbaa !0
  %arrayidx16 = getelementptr inbounds i32* %A, i64 3
  %3 = load i32* %arrayidx16, align 4, !tbaa !0
  %add20 = mul i32 %3, %mul238
  %arrayidx21 = getelementptr inbounds i32* %B, i64 3
  store i32 %add20, i32* %arrayidx21, align 4, !tbaa !0
  ret i32 0
}


; int foo_fail(int * restrict B,  int * restrict A, int n, int m) {
;   B[0] = n * A[0] + m * A[0];
;   B[1] = n * A[1] + m * A[1];
;   B[2] = n * A[2] + m * A[2];
;   B[3] = n * A[3] + m * A[3];
;   return A[0];
; }

; CHECK: @foo_fail
; CHECK-NOT: load <4 x i32>
; CHECK: ret
define i32 @foo_fail(i32* noalias nocapture %B, i32* noalias nocapture %A, i32 %n, i32 %m) #0 {
entry:
  %0 = load i32* %A, align 4, !tbaa !0
  %mul238 = add i32 %m, %n
  %add = mul i32 %0, %mul238
  store i32 %add, i32* %B, align 4, !tbaa !0
  %arrayidx4 = getelementptr inbounds i32* %A, i64 1
  %1 = load i32* %arrayidx4, align 4, !tbaa !0
  %add8 = mul i32 %1, %mul238
  %arrayidx9 = getelementptr inbounds i32* %B, i64 1
  store i32 %add8, i32* %arrayidx9, align 4, !tbaa !0
  %arrayidx10 = getelementptr inbounds i32* %A, i64 2
  %2 = load i32* %arrayidx10, align 4, !tbaa !0
  %add14 = mul i32 %2, %mul238
  %arrayidx15 = getelementptr inbounds i32* %B, i64 2
  store i32 %add14, i32* %arrayidx15, align 4, !tbaa !0
  %arrayidx16 = getelementptr inbounds i32* %A, i64 3
  %3 = load i32* %arrayidx16, align 4, !tbaa !0
  %add20 = mul i32 %3, %mul238
  %arrayidx21 = getelementptr inbounds i32* %B, i64 3
  store i32 %add20, i32* %arrayidx21, align 4, !tbaa !0
  ret i32 %0  ;<--------- This value has multiple users and can't be vectorized.
}

attributes #0 = { nounwind ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }

!0 = metadata !{metadata !"int", metadata !1}
!1 = metadata !{metadata !"omnipotent char", metadata !2}
!2 = metadata !{metadata !"Simple C/C++ TBAA"}
