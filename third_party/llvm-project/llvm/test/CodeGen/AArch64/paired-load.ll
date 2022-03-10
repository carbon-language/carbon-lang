; RUN: llc < %s | FileCheck %s
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-linux-gnu"

; Ensure we're generating ldp instructions instead of ldr Q.
; CHECK: ldp
; CHECK: stp
define void @f(i64* %p, i64* %q) {
  %addr2 = getelementptr i64, i64* %q, i32 1
  %addr = getelementptr i64, i64* %p, i32 1
  %x = load i64, i64* %p
  %y = load i64, i64* %addr
  store i64 %x, i64* %q
  store i64 %y, i64* %addr2
  ret void
}
