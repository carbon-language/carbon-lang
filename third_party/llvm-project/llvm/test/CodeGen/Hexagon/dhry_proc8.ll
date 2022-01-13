; RUN: llc -march=hexagon -O3 < %s | FileCheck %s


; Test that we generate no more than 7 packets in f0.
;
; CHECK: f0:
; CHECK: {
; CHECK: {
; CHECK: {
; CHECK: {
; CHECK: {
; CHECK: {
; CHECK: {
; CHECK-NOT: {

@g0 = external global i32

; Function Attrs: nounwind
define i32 @f0(i32* nocapture %a0, [50 x i32]* nocapture %a1, i32 %a2, i32 %a3) #0 {
b0:
  %v0 = add nsw i32 %a2, 5
  %v1 = getelementptr inbounds i32, i32* %a0, i32 %v0
  store i32 %a3, i32* %v1, align 4, !tbaa !0
  %v2 = add nsw i32 %a2, 6
  %v3 = getelementptr inbounds i32, i32* %a0, i32 %v2
  store i32 %a3, i32* %v3, align 4, !tbaa !0
  %v4 = add nsw i32 %a2, 35
  %v5 = getelementptr inbounds i32, i32* %a0, i32 %v4
  store i32 %v0, i32* %v5, align 4, !tbaa !0
  %v6 = getelementptr inbounds [50 x i32], [50 x i32]* %a1, i32 %v0, i32 %v0
  store i32 %v0, i32* %v6, align 4, !tbaa !0
  %v7 = add nsw i32 %a2, 6
  %v8 = getelementptr inbounds [50 x i32], [50 x i32]* %a1, i32 %v0, i32 %v7
  store i32 %v0, i32* %v8, align 4, !tbaa !0
  %v9 = add nsw i32 %a2, 4
  %v10 = getelementptr inbounds [50 x i32], [50 x i32]* %a1, i32 %v0, i32 %v9
  %v11 = load i32, i32* %v10, align 4, !tbaa !0
  %v12 = add nsw i32 %v11, 1
  store i32 %v12, i32* %v10, align 4, !tbaa !0
  %v13 = load i32, i32* %v1, align 4, !tbaa !0
  %v14 = add nsw i32 %a2, 25
  %v15 = getelementptr inbounds [50 x i32], [50 x i32]* %a1, i32 %v14, i32 %v0
  store i32 %v13, i32* %v15, align 4, !tbaa !0
  store i32 5, i32* @g0, align 4, !tbaa !0
  ret i32 undef
}

attributes #0 = { nounwind "target-cpu"="hexagonv60" }

!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
