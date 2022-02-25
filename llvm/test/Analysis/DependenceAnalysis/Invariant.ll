; RUN: opt < %s -disable-output "-passes=print<da>" -aa-pipeline=basic-aa 2>&1 \
; RUN: | FileCheck %s
; RUN: opt < %s -analyze -enable-new-pm=0 -basic-aa -da | FileCheck %s

; Test for a bug, which caused an assert when an invalid
; SCEVAddRecExpr is created in addToCoefficient.

; float foo (float g, float* rr[40]) {
;   float res= 0.0f;
;   for (int i = 0; i < 40; i += 5) {
;     for (int j = 0; j < 40; j += 5) {
;       float add = rr[j][j] + rr[i][j];
;       res = add > g? add : res;
;     }
;   }
;   return res;
; }

; CHECK-LABEL: foo
; CHECK: da analyze - consistent input [S 0]!
; CHECK: da analyze - input [* 0|<]!
; CHECK: da analyze - none!

define float @foo(float %g, [40 x float]* %rr) nounwind {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:
  %i.04 = phi i32 [ 0, %entry ], [ %add10, %for.inc9 ]
  %res.03 = phi float [ 0.000000e+00, %entry ], [ %add.res.1, %for.inc9 ]
  br label %for.body3

for.body3:
  %j.02 = phi i32 [ 0, %for.cond1.preheader ], [ %add8, %for.body3 ]
  %res.11 = phi float [ %res.03, %for.cond1.preheader ], [ %add.res.1, %for.body3 ]
  %arrayidx4 = getelementptr inbounds [40 x float], [40 x float]* %rr, i32 %j.02, i32 %j.02
  %0 = load float, float* %arrayidx4, align 4
  %arrayidx6 = getelementptr inbounds [40 x float], [40 x float]* %rr, i32 %i.04, i32 %j.02
  %1 = load float, float* %arrayidx6, align 4
  %add = fadd float %0, %1
  %cmp7 = fcmp ogt float %add, %g
  %add.res.1 = select i1 %cmp7, float %add, float %res.11
  %add8 = add nsw i32 %j.02, 5
  %cmp2 = icmp slt i32 %add8, 40
  br i1 %cmp2, label %for.body3, label %for.inc9

for.inc9:
  %add10 = add nsw i32 %i.04, 5
  %cmp = icmp slt i32 %add10, 40
  br i1 %cmp, label %for.cond1.preheader, label %for.end11

for.end11:
  ret float %add.res.1
}
