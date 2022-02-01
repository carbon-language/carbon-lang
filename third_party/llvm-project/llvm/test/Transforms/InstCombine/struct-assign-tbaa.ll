; RUN: opt -instcombine -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i1) nounwind

; Verify that instcombine preserves TBAA tags when converting a memcpy into
; a scalar load and store.

%struct.test1 = type { float }

; CHECK: @test
; CHECK: %[[LOAD:.*]] = load i32, i32* %{{.*}}, align 4, !tbaa !0
; CHECK: store i32 %[[LOAD:.*]], i32* %{{.*}}, align 4, !tbaa !0
; CHECK: ret
define void @test1(%struct.test1* nocapture %a, %struct.test1* nocapture %b) {
entry:
  %0 = bitcast %struct.test1* %a to i8*
  %1 = bitcast %struct.test1* %b to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %0, i8* align 4 %1, i64 4, i1 false), !tbaa.struct !3
  ret void
}

%struct.test2 = type { i32 (i8*, i32*, double*)** }

define i32 (i8*, i32*, double*)*** @test2() {
; CHECK-LABEL: @test2(
; CHECK-NOT: memcpy
; CHECK: ret
  %tmp = alloca %struct.test2, align 8
  %tmp1 = bitcast %struct.test2* %tmp to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %tmp1, i8* align 8 undef, i64 8, i1 false), !tbaa.struct !4
  %tmp2 = getelementptr %struct.test2, %struct.test2* %tmp, i32 0, i32 0
  %tmp3 = load i32 (i8*, i32*, double*)**, i32 (i8*, i32*, double*)*** %tmp2
  ret i32 (i8*, i32*, double*)*** %tmp2
}

; CHECK: !0 = !{!1, !1, i64 0}
; CHECK: !1 = !{!"float", !2}

!0 = !{!"Simple C/C++ TBAA"}
!1 = !{!"omnipotent char", !0}
!2 = !{!5, !5, i64 0}
!3 = !{i64 0, i64 4, !2}
!4 = !{i64 0, i64 8, null}
!5 = !{!"float", !0}
