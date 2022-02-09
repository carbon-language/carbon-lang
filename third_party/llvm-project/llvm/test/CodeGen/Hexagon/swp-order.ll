; RUN: llc -march=hexagon < %s -pipeliner-experimental-cg=true | FileCheck %s

; Test that when we order instructions in a packet we check for
; order dependences so that the source of an order dependence
; appears before the destination.

; CHECK: loop0(.LBB0_[[LOOP:.]],
; CHECK: .LBB0_[[LOOP]]:
; CHECK: = memw
; CHECK: = memw
; CHECK: memw({{.*}}) =
; CHECK: = memw
; CHECK: = memw
; CHECK: endloop0

@g0 = external hidden unnamed_addr constant [19 x i8], align 1

; Function Attrs: nounwind optsize
declare i32 @f0(i8* nocapture readonly, ...) #0

; Function Attrs: nounwind optsize
declare void @f1(i32*, i32*, i32* nocapture readnone) #0

; Function Attrs: argmemonly nounwind
declare i8* @llvm.hexagon.circ.stw(i8*, i32, i32, i32) #1

; Function Attrs: nounwind optsize
define void @f2(i32* %a0, i32* %a1, i32* %a2) #0 {
b0:
  %v0 = alloca i32, align 4
  call void @f1(i32* %a2, i32* %a0, i32* %v0) #2
  %v1 = bitcast i32* %a1 to i8*
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v2 = phi i32 [ 0, %b0 ], [ %v13, %b1 ]
  %v3 = phi i32* [ %a2, %b0 ], [ %v16, %b1 ]
  %v4 = phi i32 [ 0, %b0 ], [ %v14, %b1 ]
  %v5 = load i32, i32* %a1, align 4, !tbaa !0
  %v6 = add nsw i32 %v2, %v5
  %v7 = load i32, i32* %v3, align 4, !tbaa !0
  %v8 = tail call i8* @llvm.hexagon.circ.stw(i8* %v1, i32 %v7, i32 150995968, i32 4) #3
  %v9 = bitcast i8* %v8 to i32*
  %v10 = load i32, i32* %v3, align 4, !tbaa !0
  %v11 = add nsw i32 %v6, %v10
  %v12 = load i32, i32* %v9, align 4, !tbaa !0
  %v13 = add nsw i32 %v11, %v12
  %v14 = add nsw i32 %v4, 1
  %v15 = icmp eq i32 %v14, 2
  %v16 = getelementptr i32, i32* %v3, i32 1
  br i1 %v15, label %b2, label %b1

b2:                                               ; preds = %b1
  %v17 = tail call i32 (i8*, ...) @f0(i8* getelementptr inbounds ([19 x i8], [19 x i8]* @g0, i32 0, i32 0), i32 %v13) #4
  ret void
}

attributes #0 = { nounwind optsize "target-cpu"="hexagonv55" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { optsize }
attributes #3 = { nounwind }
attributes #4 = { nounwind optsize }

!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
