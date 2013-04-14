; RUN: opt < %s -basicaa -slp-vectorizer -dce -S -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32-S128"
target triple = "i386-apple-macosx10.8.0"

; int foo(double *A, int n, int m) {
;   double sum = 0, v1 = 2, v0 = 3;
;   for (int i=0; i < n; ++i)
;     sum += 7*A[i*2] + 7*A[i*2+1];
;   return sum;
; }

;CHECK: reduce
;CHECK: load <2 x double>
;CHECK: fmul <2 x double>
;CHECK: ret
define i32 @reduce(double* nocapture %A, i32 %n, i32 %m) #0 {
entry:
  %cmp13 = icmp sgt i32 %n, 0
  br i1 %cmp13, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %i.015 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %sum.014 = phi double [ %add6, %for.body ], [ 0.000000e+00, %entry ]
  %mul = shl nsw i32 %i.015, 1
  %arrayidx = getelementptr inbounds double* %A, i32 %mul
  %0 = load double* %arrayidx, align 4, !tbaa !0
  %mul1 = fmul double %0, 7.000000e+00
  %add12 = or i32 %mul, 1
  %arrayidx3 = getelementptr inbounds double* %A, i32 %add12
  %1 = load double* %arrayidx3, align 4, !tbaa !0
  %mul4 = fmul double %1, 7.000000e+00
  %add5 = fadd double %mul1, %mul4
  %add6 = fadd double %sum.014, %add5
  %inc = add nsw i32 %i.015, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %for.cond.for.end_crit_edge, label %for.body

for.cond.for.end_crit_edge:                       ; preds = %for.body
  %phitmp = fptosi double %add6 to i32
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry
  %sum.0.lcssa = phi i32 [ %phitmp, %for.cond.for.end_crit_edge ], [ 0, %entry ]
  ret i32 %sum.0.lcssa
}

attributes #0 = { nounwind readonly ssp "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }

!0 = metadata !{metadata !"double", metadata !1}
!1 = metadata !{metadata !"omnipotent char", metadata !2}
!2 = metadata !{metadata !"Simple C/C++ TBAA"}
