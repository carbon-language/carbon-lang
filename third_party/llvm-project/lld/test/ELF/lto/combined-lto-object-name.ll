; REQUIRES: x86
; RUN: llvm-as %s -o %t.o
; RUN: not ld.lld %t.o -o /dev/null 2>&1 | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

declare void @foo()
define void @_start() {
  call void @foo()
  ret void
}

; CHECK: error: undefined symbol: foo
; CHECK: >>> referenced by ld-temp.o
; CHECK:                   {{.*}}:(_start)
