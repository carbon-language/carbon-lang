; RUN: opt -S -tbaa -basicaa -memcpyopt -instcombine < %s | FileCheck %s

target datalayout = "e-p:64:64:64"

; The second memcpy is redundant and can be deleted. There's an intervening store, but
; it has a TBAA tag which declares that it is unrelated.

; CHECK: @foo
; CHECK-NEXT: tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %p, i8* %q, i64 16, i32 1, i1 false), !tbaa !0
; CHECK-NEXT: store i8 2, i8* %s, align 1, !tbaa [[TAGA:!.*]]
; CHECK-NEXT: ret void
define void @foo(i8* nocapture %p, i8* nocapture %q, i8* nocapture %s) nounwind {
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %p, i8* %q, i64 16, i32 1, i1 false), !tbaa !2
  store i8 2, i8* %s, align 1, !tbaa !1
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %q, i8* %p, i64 16, i32 1, i1 false), !tbaa !2
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i32, i1) nounwind

; CHECK [[TAGA]] = metadata !{metadata [[TYPEA:!.*]], metadata [[TYPEA]], i64 0}
; CHECK [[TYPEA]] = metadata !{metadata !"A", metadata !{{.*}}}
!0 = metadata !{metadata !"tbaa root", null}
!1 = metadata !{metadata !3, metadata !3, i64 0}
!2 = metadata !{metadata !4, metadata !4, i64 0}
!3 = metadata !{metadata !"A", metadata !0}
!4 = metadata !{metadata !"B", metadata !0}
