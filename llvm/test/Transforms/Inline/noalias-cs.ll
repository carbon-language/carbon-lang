; RUN: opt -inline -enable-noalias-to-md-conversion -S < %s | FileCheck %s
target datalayout = "e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define void @foo2(float* nocapture %a, float* nocapture %b, float* nocapture readonly %c) #0 {
entry:
  %0 = load float* %c, align 4, !noalias !3
  %arrayidx.i = getelementptr inbounds float* %a, i64 5
  store float %0, float* %arrayidx.i, align 4, !alias.scope !7, !noalias !8
  %arrayidx1.i = getelementptr inbounds float* %b, i64 8
  store float %0, float* %arrayidx1.i, align 4, !alias.scope !8, !noalias !7
  %1 = load float* %c, align 4
  %arrayidx = getelementptr inbounds float* %a, i64 7
  store float %1, float* %arrayidx, align 4
  ret void
}

define void @foo(float* nocapture %a, float* nocapture %b, float* nocapture readonly %c) #0 {
entry:
  call void @foo2(float* %a, float* %b, float* %c), !noalias !0
  call void @foo2(float* %b, float* %b, float* %a), !alias.scope !0
  ret void
}

; CHECK: define void @foo(float* nocapture %a, float* nocapture %b, float* nocapture readonly %c) #0 {
; CHECK: entry:
; CHECK:   %0 = load float* %c, align 4, !noalias !6
; CHECK:   %arrayidx.i.i = getelementptr inbounds float* %a, i64 5
; CHECK:   store float %0, float* %arrayidx.i.i, align 4, !alias.scope !12, !noalias !13
; CHECK:   %arrayidx1.i.i = getelementptr inbounds float* %b, i64 8
; CHECK:   store float %0, float* %arrayidx1.i.i, align 4, !alias.scope !14, !noalias !15
; CHECK:   %1 = load float* %c, align 4, !noalias !16
; CHECK:   %arrayidx.i = getelementptr inbounds float* %a, i64 7
; CHECK:   store float %1, float* %arrayidx.i, align 4, !noalias !16
; CHECK:   %2 = load float* %a, align 4, !alias.scope !16, !noalias !17
; CHECK:   %arrayidx.i.i1 = getelementptr inbounds float* %b, i64 5
; CHECK:   store float %2, float* %arrayidx.i.i1, align 4, !alias.scope !21, !noalias !22
; CHECK:   %arrayidx1.i.i2 = getelementptr inbounds float* %b, i64 8
; CHECK:   store float %2, float* %arrayidx1.i.i2, align 4, !alias.scope !23, !noalias !24
; CHECK:   %3 = load float* %a, align 4, !alias.scope !16
; CHECK:   %arrayidx.i3 = getelementptr inbounds float* %b, i64 7
; CHECK:   store float %3, float* %arrayidx.i3, align 4, !alias.scope !16
; CHECK:   ret void
; CHECK: }

attributes #0 = { nounwind uwtable }

!0 = metadata !{metadata !1}
!1 = metadata !{metadata !1, metadata !2, metadata !"hello: %a"}
!2 = metadata !{metadata !2, metadata !"hello"}
!3 = metadata !{metadata !4, metadata !6}
!4 = metadata !{metadata !4, metadata !5, metadata !"hello2: %a"}
!5 = metadata !{metadata !5, metadata !"hello2"}
!6 = metadata !{metadata !6, metadata !5, metadata !"hello2: %b"}
!7 = metadata !{metadata !4}
!8 = metadata !{metadata !6}

; CHECK: !0 = metadata !{metadata !1, metadata !3}
; CHECK: !1 = metadata !{metadata !1, metadata !2, metadata !"hello2: %a"}
; CHECK: !2 = metadata !{metadata !2, metadata !"hello2"}
; CHECK: !3 = metadata !{metadata !3, metadata !2, metadata !"hello2: %b"}
; CHECK: !4 = metadata !{metadata !1}
; CHECK: !5 = metadata !{metadata !3}
; CHECK: !6 = metadata !{metadata !7, metadata !9, metadata !10}
; CHECK: !7 = metadata !{metadata !7, metadata !8, metadata !"hello2: %a"}
; CHECK: !8 = metadata !{metadata !8, metadata !"hello2"}
; CHECK: !9 = metadata !{metadata !9, metadata !8, metadata !"hello2: %b"}
; CHECK: !10 = metadata !{metadata !10, metadata !11, metadata !"hello: %a"}
; CHECK: !11 = metadata !{metadata !11, metadata !"hello"}
; CHECK: !12 = metadata !{metadata !7}
; CHECK: !13 = metadata !{metadata !9, metadata !10}
; CHECK: !14 = metadata !{metadata !9}
; CHECK: !15 = metadata !{metadata !7, metadata !10}
; CHECK: !16 = metadata !{metadata !10}
; CHECK: !17 = metadata !{metadata !18, metadata !20}
; CHECK: !18 = metadata !{metadata !18, metadata !19, metadata !"hello2: %a"}
; CHECK: !19 = metadata !{metadata !19, metadata !"hello2"}
; CHECK: !20 = metadata !{metadata !20, metadata !19, metadata !"hello2: %b"}
; CHECK: !21 = metadata !{metadata !18, metadata !10}
; CHECK: !22 = metadata !{metadata !20}
; CHECK: !23 = metadata !{metadata !20, metadata !10}
; CHECK: !24 = metadata !{metadata !18}

