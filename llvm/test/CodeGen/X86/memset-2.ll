; RUN: llc -mtriple=i386-apple-darwin < %s | FileCheck %s

declare void @llvm.memset.i32(i8*, i8, i32, i32) nounwind

define fastcc void @t1() nounwind {
entry:
; CHECK: t1:
; CHECK: call _memset
  call void @llvm.memset.i32( i8* null, i8 0, i32 188, i32 1 ) nounwind
  unreachable
}

define fastcc void @t2(i8 signext %c) nounwind {
entry:
; CHECK: t2:
; CHECK: call _memset
  call void @llvm.memset.i32( i8* undef, i8 %c, i32 76, i32 1 ) nounwind
  unreachable
}
