; RUN: opt < %s -basicaa -scoped-noalias -aa-eval -evaluate-aa-metadata -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo2(float* nocapture %a, float* nocapture %b, float* nocapture readonly %c) #0 {
entry:
; CHECK-LABEL: Function: foo2
  %0 = load float* %c, align 4, !alias.scope !0
  %arrayidx.i = getelementptr inbounds float* %a, i64 5
  store float %0, float* %arrayidx.i, align 4, !alias.scope !2, !noalias !1
  %arrayidx1.i = getelementptr inbounds float* %b, i64 8
  store float %0, float* %arrayidx1.i, align 4, !alias.scope !1, !noalias !2
  %1 = load float* %c, align 4
  %arrayidx = getelementptr inbounds float* %a, i64 7
  store float %1, float* %arrayidx, align 4
  ret void

; CHECK: NoAlias:   %0 = load float* %c, align 4, !alias.scope !0 <->   store float %0, float* %arrayidx.i, align 4, !alias.scope !2, !noalias !1
; CHECK: NoAlias:   %0 = load float* %c, align 4, !alias.scope !0 <->   store float %0, float* %arrayidx1.i, align 4, !alias.scope !1, !noalias !2
; CHECK: MayAlias:   %0 = load float* %c, align 4, !alias.scope !0 <->   store float %1, float* %arrayidx, align 4
; CHECK: MayAlias:   %1 = load float* %c, align 4 <->   store float %0, float* %arrayidx.i, align 4, !alias.scope !2, !noalias !1
; CHECK: MayAlias:   %1 = load float* %c, align 4 <->   store float %0, float* %arrayidx1.i, align 4, !alias.scope !1, !noalias !2
; CHECK: MayAlias:   %1 = load float* %c, align 4 <->   store float %1, float* %arrayidx, align 4
; CHECK: NoAlias:   store float %0, float* %arrayidx1.i, align 4, !alias.scope !1, !noalias !2 <->   store float %0, float* %arrayidx.i, align 4, !alias.scope !2, !noalias !1
; CHECK: NoAlias:   store float %1, float* %arrayidx, align 4 <->   store float %0, float* %arrayidx.i, align 4, !alias.scope !2, !noalias !1
; CHECK: MayAlias:   store float %1, float* %arrayidx, align 4 <->   store float %0, float* %arrayidx1.i, align 4, !alias.scope !1, !noalias !2
}

attributes #0 = { nounwind uwtable }

!0 = metadata !{metadata !1, metadata !2}
!1 = metadata !{metadata !1}
!2 = metadata !{metadata !2}

