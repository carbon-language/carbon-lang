; RUN: llc < %s | FileCheck %s

target triple = "i386"

declare void @llvm.memset.i32(i8*, i8, i32, i32) nounwind

define fastcc void @t() nounwind {
entry:
; CHECK: t:
; CHECK: call memset
  call void @llvm.memset.i32( i8* null, i8 0, i32 188, i32 1 ) nounwind
  unreachable
}
