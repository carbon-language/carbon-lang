; RUN: opt -S -analyze -stack-safety-local < %s | FileCheck %s
; RUN: opt -S -passes="print<stack-safety-local>" -disable-output < %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: 'Stack Safety Local Analysis' for function 'Foo'
; CHECK-NEXT: Not Implemented

define dso_local void @Foo() {
entry:
  ret void
}
