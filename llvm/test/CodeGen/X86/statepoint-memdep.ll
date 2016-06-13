; RUN: opt -S -dse < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @f() {
  ; CHECK-LABEL: @f(
  %s = alloca i64
  ; Verify that this first store is not considered killed by the second one
  ; since it could be observed from the deopt continuation.
  ; CHECK: store i64 1, i64* %s
  store i64 1, i64* %s
  call void @g() [ "deopt"(i64* %s) ]
  store i64 0, i64* %s
  ret void
}

declare void @g()
