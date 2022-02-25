; RUN: opt -module-summary %s -o %t.o
; RUN: llvm-lto2 run -o %t1.o %t.o -r %t.o,patatino,pr
; RUN: llvm-readobj --symbols %t1.o.1 | FileCheck %s

; CHECK: Name: patatino
; CHECK-NEXT: Value:
; CHECK-NEXT: Size:
; CHECK-NEXT: Binding: Weak
; CHECK-NEXT: Type: Function

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @patatino() {
  ret void
}
