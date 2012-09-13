; RUN: opt -instcombine -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

%struct.foo = type { float }

; Verify that instcombine preserves TBAA tags when converting a memcpy into
; a scalar load and store.

; CHECK: %2 = load float* %0, align 4, !tbaa !0
; CHECK: store float %2, float* %1, align 4, !tbaa !0
; CHECK: !0 = metadata !{metadata !"float", metadata !1}
define void @test(%struct.foo* nocapture %a, %struct.foo* nocapture %b) {
entry:
  %0 = bitcast %struct.foo* %a to i8*
  %1 = bitcast %struct.foo* %b to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %0, i8* %1, i64 4, i32 4, i1 false), !tbaa.struct !3
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i32, i1) nounwind

!0 = metadata !{metadata !"Simple C/C++ TBAA"}
!1 = metadata !{metadata !"omnipotent char", metadata !0}
!2 = metadata !{metadata !"float", metadata !0}
!3 = metadata !{i64 0, i64 4, metadata !2}
