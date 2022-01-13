; RUN: opt -basic-aa -print-memoryssa -verify-memoryssa -enable-new-pm=0 -analyze < %s 2>&1 | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa -passes='print<memoryssa>,verify<memoryssa>' -disable-output < %s 2>&1 | FileCheck %s
; This test checks that lifetime markers are considered clobbers of %P,
; and due to lack of noalias information, of %Q as well.

define i8 @test(i8* %P, i8* %Q) {
entry:
; CHECK:  1 = MemoryDef(liveOnEntry)
; CHECK-NEXT:   call void @llvm.lifetime.start.p0i8(i64 32, i8* %P)
  call void @llvm.lifetime.start.p0i8(i64 32, i8* %P)
; CHECK:  MemoryUse(1)
; CHECK-NEXT:   %0 = load i8, i8* %P
  %0 = load i8, i8* %P
; CHECK:  2 = MemoryDef(1)
; CHECK-NEXT:   store i8 1, i8* %P
  store i8 1, i8* %P
; CHECK:  3 = MemoryDef(2)
; CHECK-NEXT:   call void @llvm.lifetime.end.p0i8(i64 32, i8* %P)
  call void @llvm.lifetime.end.p0i8(i64 32, i8* %P)
; CHECK:  MemoryUse(3)
; CHECK-NEXT:   %1 = load i8, i8* %P
  %1 = load i8, i8* %P
; CHECK:  MemoryUse(3)
; CHECK-NEXT:   %2 = load i8, i8* %Q
  %2 = load i8, i8* %Q
  ret i8 %1
}
declare void @llvm.lifetime.start.p0i8(i64 %S, i8* nocapture %P) readonly
declare void @llvm.lifetime.end.p0i8(i64 %S, i8* nocapture %P)
