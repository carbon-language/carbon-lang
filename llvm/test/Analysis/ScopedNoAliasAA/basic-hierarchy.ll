; RUN: opt < %s -basicaa -scoped-noalias -aa-eval -evaluate-aa-metadata -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo1(float* nocapture %a, float* nocapture readonly %c) #0 {
entry:
; CHECK-LABEL: Function: foo1
  %0 = load float* %c, align 4, !alias.scope !0
  %arrayidx.i = getelementptr inbounds float* %a, i64 5
  store float %0, float* %arrayidx.i, align 4, !noalias !0

  %1 = load float* %c, align 4
  %arrayidx = getelementptr inbounds float* %a, i64 7
  store float %1, float* %arrayidx, align 4

  %2 = load float* %c, align 4, !alias.scope !1
  %arrayidx.i2 = getelementptr inbounds float* %a, i64 15
  store float %2, float* %arrayidx.i2, align 4, !noalias !3

  %3 = load float* %c, align 4, !alias.scope !3
  %arrayidx.i3 = getelementptr inbounds float* %a, i64 16
  store float %3, float* %arrayidx.i3, align 4, !noalias !0

  %4 = load float* %c, align 4, !alias.scope !5
  %arrayidx.i4 = getelementptr inbounds float* %a, i64 17
  store float %4, float* %arrayidx.i4, align 4, !noalias !3
  ret void
}

attributes #0 = { nounwind uwtable }

; A root scope (which doubles as a list of itself):
!0 = metadata !{metadata !0}

; Two child scopes (which must be self-referential to avoid being "uniqued"):
!1 = metadata !{metadata !2}
!2 = metadata !{metadata !2, metadata !0}

!3 = metadata !{metadata !4}
!4 = metadata !{metadata !4, metadata !0}

; A list of the two children:
!5 = metadata !{metadata !2, metadata !4}

; CHECK: NoAlias:   %0 = load float* %c, align 4, !alias.scope !0 <->   store float %0, float* %arrayidx.i, align 4, !noalias !0
; CHECK: MayAlias:   %0 = load float* %c, align 4, !alias.scope !0 <->   store float %1, float* %arrayidx, align 4
; CHECK: MayAlias:   %0 = load float* %c, align 4, !alias.scope !0 <->   store float %2, float* %arrayidx.i2, align 4, !noalias !3
; CHECK: NoAlias:   %0 = load float* %c, align 4, !alias.scope !0 <->   store float %3, float* %arrayidx.i3, align 4, !noalias !0
; CHECK: MayAlias:   %0 = load float* %c, align 4, !alias.scope !0 <->   store float %4, float* %arrayidx.i4, align 4, !noalias !3
; CHECK: MayAlias:   %1 = load float* %c, align 4 <->   store float %0, float* %arrayidx.i, align 4, !noalias !0
; CHECK: MayAlias:   %1 = load float* %c, align 4 <->   store float %1, float* %arrayidx, align 4
; CHECK: MayAlias:   %1 = load float* %c, align 4 <->   store float %2, float* %arrayidx.i2, align 4, !noalias !3
; CHECK: MayAlias:   %1 = load float* %c, align 4 <->   store float %3, float* %arrayidx.i3, align 4, !noalias !0
; CHECK: MayAlias:   %1 = load float* %c, align 4 <->   store float %4, float* %arrayidx.i4, align 4, !noalias !3
; CHECK: NoAlias:   %2 = load float* %c, align 4, !alias.scope !1 <->   store float %0, float* %arrayidx.i, align 4, !noalias !0
; CHECK: MayAlias:   %2 = load float* %c, align 4, !alias.scope !1 <->   store float %1, float* %arrayidx, align 4
; CHECK: MayAlias:   %2 = load float* %c, align 4, !alias.scope !1 <->   store float %2, float* %arrayidx.i2, align 4, !noalias !3
; CHECK: NoAlias:   %2 = load float* %c, align 4, !alias.scope !1 <->   store float %3, float* %arrayidx.i3, align 4, !noalias !0
; CHECK: MayAlias:   %2 = load float* %c, align 4, !alias.scope !1 <->   store float %4, float* %arrayidx.i4, align 4, !noalias !3
; CHECK: NoAlias:   %3 = load float* %c, align 4, !alias.scope !3 <->   store float %0, float* %arrayidx.i, align 4, !noalias !0
; CHECK: MayAlias:   %3 = load float* %c, align 4, !alias.scope !3 <->   store float %1, float* %arrayidx, align 4
; CHECK: NoAlias:   %3 = load float* %c, align 4, !alias.scope !3 <->   store float %2, float* %arrayidx.i2, align 4, !noalias !3
; CHECK: NoAlias:   %3 = load float* %c, align 4, !alias.scope !3 <->   store float %3, float* %arrayidx.i3, align 4, !noalias !0
; CHECK: NoAlias:   %3 = load float* %c, align 4, !alias.scope !3 <->   store float %4, float* %arrayidx.i4, align 4, !noalias !3
; CHECK: NoAlias:   %4 = load float* %c, align 4, !alias.scope !5 <->   store float %0, float* %arrayidx.i, align 4, !noalias !0
; CHECK: MayAlias:   %4 = load float* %c, align 4, !alias.scope !5 <->   store float %1, float* %arrayidx, align 4
; CHECK: NoAlias:   %4 = load float* %c, align 4, !alias.scope !5 <->   store float %2, float* %arrayidx.i2, align 4, !noalias !3
; CHECK: NoAlias:   %4 = load float* %c, align 4, !alias.scope !5 <->   store float %3, float* %arrayidx.i3, align 4, !noalias !0
; CHECK: NoAlias:   %4 = load float* %c, align 4, !alias.scope !5 <->   store float %4, float* %arrayidx.i4, align 4, !noalias !3
; CHECK: NoAlias:   store float %1, float* %arrayidx, align 4 <->   store float %0, float* %arrayidx.i, align 4, !noalias !0
; CHECK: NoAlias:   store float %2, float* %arrayidx.i2, align 4, !noalias !3 <->   store float %0, float* %arrayidx.i, align 4, !noalias !0
; CHECK: NoAlias:   store float %2, float* %arrayidx.i2, align 4, !noalias !3 <->   store float %1, float* %arrayidx, align 4
; CHECK: NoAlias:   store float %3, float* %arrayidx.i3, align 4, !noalias !0 <->   store float %0, float* %arrayidx.i, align 4, !noalias !0
; CHECK: NoAlias:   store float %3, float* %arrayidx.i3, align 4, !noalias !0 <->   store float %1, float* %arrayidx, align 4
; CHECK: NoAlias:   store float %3, float* %arrayidx.i3, align 4, !noalias !0 <->   store float %2, float* %arrayidx.i2, align 4, !noalias !3
; CHECK: NoAlias:   store float %4, float* %arrayidx.i4, align 4, !noalias !3 <->   store float %0, float* %arrayidx.i, align 4, !noalias !0
; CHECK: NoAlias:   store float %4, float* %arrayidx.i4, align 4, !noalias !3 <->   store float %1, float* %arrayidx, align 4
; CHECK: NoAlias:   store float %4, float* %arrayidx.i4, align 4, !noalias !3 <->   store float %2, float* %arrayidx.i2, align 4, !noalias !3
; CHECK: NoAlias:   store float %4, float* %arrayidx.i4, align 4, !noalias !3 <->   store float %3, float* %arrayidx.i3, align 4, !noalias !0

