; RUN: llc -march=hexagon < %s | FileCheck %s
target triple = "hexagon"

; Function Attrs: norecurse nounwind
define void @test(<32 x i32>* nocapture readonly %x, <32 x i32>* nocapture readnone %y, <32 x i32>* nocapture %a, <32 x i32>* nocapture %b) #0 {
entry:
; CHECK: v0 = vmem(r0+#7):nt
  %add.ptr = getelementptr inbounds <32 x i32>, <32 x i32>* %x, i32 7
  %0 = load <32 x i32>, <32 x i32>* %add.ptr, align 128, !tbaa !1, !nontemporal !4

; CHECK: v1.cur = vmem(r2+#0):nt
  %1 = load <32 x i32>, <32 x i32>* %a, align 128, !tbaa !1, !nontemporal !4

; CHECK: vmem(r3+#3):nt = v1
  %add.ptr2 = getelementptr inbounds <32 x i32>, <32 x i32>* %b, i32 3
  store <32 x i32> %1, <32 x i32>* %add.ptr2, align 128, !tbaa !1, !nontemporal !4

; CHECK: vmem(r2+#0):nt = v0
  store <32 x i32> %0, <32 x i32>* %a, align 128, !tbaa !1, !nontemporal !4
  ret void
}

attributes #0 = { norecurse nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length128b" }

!1 = !{!2, !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{i32 1}
