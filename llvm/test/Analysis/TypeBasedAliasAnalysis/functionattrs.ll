; RUN: opt < %s -enable-tbaa -tbaa -basicaa -functionattrs -S | FileCheck %s

; FunctionAttrs should make use of TBAA.

; CHECK: define void @test0_yes(i32* nocapture %p) nounwind readnone {
define void @test0_yes(i32* %p) nounwind {
  store i32 0, i32* %p, !tbaa !1
  ret void
}

; CHECK: define void @test0_no(i32* nocapture %p) nounwind {
define void @test0_no(i32* %p) nounwind {
  store i32 0, i32* %p, !tbaa !2
  ret void
}

; CHECK: define void @test1_yes(i32* %p) nounwind readonly {
define void @test1_yes(i32* %p) nounwind {
  call void @callee(i32* %p), !tbaa !1
  ret void
}

; CHECK: define void @test1_no(i32* %p) nounwind {
define void @test1_no(i32* %p) nounwind {
  call void @callee(i32* %p), !tbaa !2
  ret void
}

declare void @callee(i32* %p) nounwind

; Root note.
!0 = metadata !{ }

; Invariant memory.
!1 = metadata !{ metadata !"foo", metadata !0, i1 1 }
; Not invariant memory.
!2 = metadata !{ metadata !"foo", metadata !0, i1 0 }
