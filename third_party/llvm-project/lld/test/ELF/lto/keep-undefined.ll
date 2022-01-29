; REQUIRES: x86
; This test checks that symbols which are specified in "-u" switches
; are kept over LTO if we link an executable.
; RUN: llvm-as %s -o %t.o
; RUN: ld.lld %t.o -o %tout -u foo
; RUN: llvm-nm %tout | FileCheck %s

; CHECK: T foo

target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @foo() {
  ret void
}

define void @_start() {
  call void @foo()
  ret void
}
