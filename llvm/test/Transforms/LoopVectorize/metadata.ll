; RUN: opt < %s  -loop-vectorize -force-vector-unroll=1 -force-vector-width=4 -S | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define i32 @test1(i32* nocapture %a, float* nocapture readonly %b) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float* %b, i64 %indvars.iv
  %0 = load float* %arrayidx, align 4, !tbaa !0
  %conv = fptosi float %0 to i32
  %arrayidx2 = getelementptr inbounds i32* %a, i64 %indvars.iv
  store i32 %conv, i32* %arrayidx2, align 4, !tbaa !4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1600
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret i32 0
}

; CHECK-LABEL: @test1
; CHECK: load <4 x float>* %{{.*}}, align 4, !tbaa ![[TFLT:[0-9]+]]
; CHECK: store <4 x i32> %{{.*}}, <4 x i32>* %{{.*}}, align 4, !tbaa ![[TINT:[0-9]+]]
; CHECK: ret i32 0

; CHECK-DAG: ![[TFLT]] = metadata !{metadata ![[TFLT1:[0-9]+]]
; CHECK-DAG: ![[TFLT1]] = metadata !{metadata !"float"

; CHECK-DAG: ![[TINT]] = metadata !{metadata ![[TINT1:[0-9]+]]
; CHECK-DAG: ![[TINT1]] = metadata !{metadata !"int"

attributes #0 = { nounwind uwtable }

!0 = metadata !{metadata !1, metadata !1, i64 0}
!1 = metadata !{metadata !"float", metadata !2, i64 0}
!2 = metadata !{metadata !"omnipotent char", metadata !3, i64 0}
!3 = metadata !{metadata !"Simple C/C++ TBAA"}
!4 = metadata !{metadata !5, metadata !5, i64 0}
!5 = metadata !{metadata !"int", metadata !2, i64 0}

