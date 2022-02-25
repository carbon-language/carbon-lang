; RUN: opt -tbaa -basic-aa -gvn -S < %s | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:32:64-v128:32:128-a0:0:32-n32"

; TBAA should prove that these calls don't interfere, since they are
; IntrArgReadMem and have TBAA metadata.

; CHECK:      define <8 x i16> @test0(<8 x i16>* %p, <8 x i16>* %q, <8 x i16> %y, <8 x i1> %m, <8 x i16> %pt) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %a = call <8 x i16> @llvm.masked.load.v8i16.p0v8i16(<8 x i16>* %p, i32 16, <8 x i1> %m, <8 x i16> %pt) [[NUW:#[0-9]+]]
; CHECK-NEXT:   call void @llvm.masked.store.v8i16.p0v8i16(<8 x i16> %y, <8 x i16>* %q, i32 16, <8 x i1> %m)
; CHECK-NEXT:   %c = add <8 x i16> %a, %a
define <8 x i16> @test0(<8 x i16>* %p, <8 x i16>* %q, <8 x i16> %y, <8 x i1> %m, <8 x i16> %pt) {
entry:
  %a = call <8 x i16> @llvm.masked.load.v8i16.p0v8i16(<8 x i16>* %p, i32 16, <8 x i1> %m, <8 x i16> %pt) nounwind, !tbaa !2
  call void @llvm.masked.store.v8i16.p0v8i16(<8 x i16> %y, <8 x i16>* %q, i32 16, <8 x i1> %m), !tbaa !1
  %b = call <8 x i16> @llvm.masked.load.v8i16.p0v8i16(<8 x i16>* %p, i32 16, <8 x i1> %m, <8 x i16> %pt) nounwind, !tbaa !2
  %c = add <8 x i16> %a, %b
  ret <8 x i16> %c
}

declare <8 x i16> @llvm.masked.load.v8i16.p0v8i16(<8 x i16>*, i32, <8 x i1>, <8 x i16>) nounwind readonly
declare void @llvm.masked.store.v8i16.p0v8i16(<8 x i16>, <8 x i16>*, i32, <8 x i1>) nounwind

; CHECK: attributes #0 = { argmemonly nofree nosync nounwind readonly willreturn }
; CHECK: attributes #1 = { argmemonly nofree nosync nounwind willreturn writeonly }
; CHECK: attributes [[NUW]] = { nounwind }

!0 = !{!"tbaa root"}
!1 = !{!3, !3, i64 0}
!2 = !{!4, !4, i64 0}
!3 = !{!"A", !0}
!4 = !{!"B", !0}
