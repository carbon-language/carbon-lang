; RUN: opt -inline -enable-noalias-to-md-conversion -S < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @hello(float* noalias nocapture %a, float* nocapture readonly %c) #0 {
entry:
  %0 = load float* %c, align 4
  %arrayidx = getelementptr inbounds float* %a, i64 5
  store float %0, float* %arrayidx, align 4
  ret void
}

define void @foo(float* nocapture %a, float* nocapture readonly %c) #0 {
entry:
  tail call void @hello(float* %a, float* %c)
  %0 = load float* %c, align 4
  %arrayidx = getelementptr inbounds float* %a, i64 7
  store float %0, float* %arrayidx, align 4
  ret void
}

; CHECK: define void @foo(float* nocapture %a, float* nocapture readonly %c) #0 {
; CHECK: entry:
; CHECK:   %0 = load float* %c, align 4, !noalias !0
; CHECK:   %arrayidx.i = getelementptr inbounds float* %a, i64 5
; CHECK:   store float %0, float* %arrayidx.i, align 4, !alias.scope !0
; CHECK:   %1 = load float* %c, align 4
; CHECK:   %arrayidx = getelementptr inbounds float* %a, i64 7
; CHECK:   store float %1, float* %arrayidx, align 4
; CHECK:   ret void
; CHECK: }

define void @hello2(float* noalias nocapture %a, float* noalias nocapture %b, float* nocapture readonly %c) #0 {
entry:
  %0 = load float* %c, align 4
  %arrayidx = getelementptr inbounds float* %a, i64 5
  store float %0, float* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds float* %b, i64 8
  store float %0, float* %arrayidx1, align 4
  ret void
}

define void @foo2(float* nocapture %a, float* nocapture %b, float* nocapture readonly %c) #0 {
entry:
  tail call void @hello2(float* %a, float* %b, float* %c)
  %0 = load float* %c, align 4
  %arrayidx = getelementptr inbounds float* %a, i64 7
  store float %0, float* %arrayidx, align 4
  ret void
}

; CHECK: define void @foo2(float* nocapture %a, float* nocapture %b, float* nocapture readonly %c) #0 {
; CHECK: entry:
; CHECK:   %0 = load float* %c, align 4, !noalias !3
; CHECK:   %arrayidx.i = getelementptr inbounds float* %a, i64 5
; CHECK:   store float %0, float* %arrayidx.i, align 4, !alias.scope !7, !noalias !8
; CHECK:   %arrayidx1.i = getelementptr inbounds float* %b, i64 8
; CHECK:   store float %0, float* %arrayidx1.i, align 4, !alias.scope !8, !noalias !7
; CHECK:   %1 = load float* %c, align 4
; CHECK:   %arrayidx = getelementptr inbounds float* %a, i64 7
; CHECK:   store float %1, float* %arrayidx, align 4
; CHECK:   ret void
; CHECK: }

attributes #0 = { nounwind uwtable }

; CHECK: !0 = metadata !{metadata !1}
; CHECK: !1 = metadata !{metadata !1, metadata !2, metadata !"hello: %a"}
; CHECK: !2 = metadata !{metadata !2, metadata !"hello"}
; CHECK: !3 = metadata !{metadata !4, metadata !6}
; CHECK: !4 = metadata !{metadata !4, metadata !5, metadata !"hello2: %a"}
; CHECK: !5 = metadata !{metadata !5, metadata !"hello2"}
; CHECK: !6 = metadata !{metadata !6, metadata !5, metadata !"hello2: %b"}
; CHECK: !7 = metadata !{metadata !4}
; CHECK: !8 = metadata !{metadata !6}

