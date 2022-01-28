; RUN: llc -march=hexagon < %s | FileCheck %s

; Test that we generate a post-increment when using double hvx (128B)
; post-increment operations.

; CHECK: = vmem(r{{[0-9]+}}++#1)
; CHECK: vmem(r{{[0-9]+}}++#1)

; Function Attrs: nounwind
define void @f0(i8* noalias nocapture readonly %a0, i8* noalias nocapture %a1, i32 %a2) #0 {
b0:
  %v0 = icmp sgt i32 %a2, 0
  br i1 %v0, label %b1, label %b3

b1:                                               ; preds = %b0
  %v1 = bitcast i8* %a0 to <32 x i32>*
  %v2 = bitcast i8* %a1 to <32 x i32>*
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v3 = phi <32 x i32>* [ %v9, %b2 ], [ %v1, %b1 ]
  %v4 = phi <32 x i32>* [ %v10, %b2 ], [ %v2, %b1 ]
  %v5 = phi i32 [ %v7, %b2 ], [ 0, %b1 ]
  %v6 = load <32 x i32>, <32 x i32>* %v3, align 128, !tbaa !0
  store <32 x i32> %v6, <32 x i32>* %v4, align 128, !tbaa !0
  %v7 = add nsw i32 %v5, 1
  %v8 = icmp eq i32 %v7, %a2
  %v9 = getelementptr <32 x i32>, <32 x i32>* %v3, i32 1
  %v10 = getelementptr <32 x i32>, <32 x i32>* %v4, i32 1
  br i1 %v8, label %b3, label %b2

b3:                                               ; preds = %b2, %b0
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length128b" }

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2, i64 0}
!2 = !{!"Simple C/C++ TBAA"}
