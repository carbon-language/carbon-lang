; REQUIRES: x86
; RUN: llvm-as %s -o %t.o
; RUN: llvm-as %p/Inputs/type-merge2.ll -o %t2.o
; RUN: ld.lld %t.o %t2.o -o %t.so -shared -save-temps
; RUN: llvm-dis %t.so.0.0.preopt.bc -o - | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

%zed = type { i8 }
define void @foo()  {
  call void @bar(%zed* null)
  ret void
}
declare void @bar(%zed*)

; CHECK:      define void @foo() {
; CHECK-NEXT:   call void @bar(ptr null)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK:      define void @bar(ptr %this) {
; CHECK-NEXT:   store ptr %this, ptr null
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
