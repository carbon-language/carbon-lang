; RUN: llc -march=hexagon -O2 < %s | FileCheck %s
; CHECK: mem{{.*}} = {{.*}}.new

target triple = "hexagon-unknown-linux-gnu"

; Function Attrs: nounwind
define void @f0(i8* nocapture %a0, i8* nocapture %a1) #0 {
b0:
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v0 = phi i8* [ %a1, %b0 ], [ %v2, %b1 ]
  %v1 = phi i8* [ %a0, %b0 ], [ %v4, %b1 ]
  %v2 = getelementptr inbounds i8, i8* %v0, i32 1
  %v3 = load volatile i8, i8* %v0, align 1, !tbaa !0
  %v4 = getelementptr inbounds i8, i8* %v1, i32 1
  store volatile i8 %v3, i8* %v1, align 1, !tbaa !0
  %v5 = icmp eq i8 %v3, 0
  br i1 %v5, label %b2, label %b1

b2:                                               ; preds = %b1
  ret void
}

attributes #0 = { nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
