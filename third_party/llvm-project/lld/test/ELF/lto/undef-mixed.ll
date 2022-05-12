; REQUIRES: x86
; RUN: llvm-mc %p/Inputs/undef-mixed.s -o %t.o -filetype=obj -triple=x86_64-pc-linux
; RUN: llvm-as %s -o %t2.o
; RUN: ld.lld %t2.o %t.o -o %t.so -shared
; RUN: llvm-readobj --symbols %t.so | FileCheck %s

; CHECK:      Name: bar
; CHECK-NEXT: Value:
; CHECK-NEXT: Size: 0
; CHECK-NEXT: Binding: Global
; CHECK-NEXT: Type: None
; CHECK-NEXT: Other: 0
; CHECK-NEXT: Section: .text

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @bar()
define void @foo() {
  call void @bar()
  ret void
}
