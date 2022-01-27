; RUN: llc -march=hexagon -O2 < %s | FileCheck %s
; CHECK: dcfetch(

target triple = "hexagon"

; Function Attrs: nounwind
define i32 @f0() #0 {
b0:
  %v0 = alloca i32, align 4
  store i32 0, i32* %v0, align 4, !tbaa !0
  %v1 = bitcast i32* %v0 to i8*
  call void @llvm.hexagon.prefetch(i8* %v1)
  ret i32 0
}

; Function Attrs: nounwind
declare void @llvm.hexagon.prefetch(i8*) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" }
attributes #1 = { nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
