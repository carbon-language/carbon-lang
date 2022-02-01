; REQUIRES: x86
; RUN: llvm-as %s -o %t1.o
; RUN: llvm-as %p/Inputs/available-externally.ll -o %t2.o
; RUN: ld.lld %t1.o %t2.o -o %t.so -shared -save-temps
; RUN: llvm-dis < %t.so.0.2.internalize.bc | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo() {
  call void @bar()
  call void @zed()
  ret void
}
define available_externally void @bar() {
  ret void
}
define available_externally void @zed() {
  ret void
}

; CHECK: define available_externally void @bar() {
; CHECK: define void @zed() {
