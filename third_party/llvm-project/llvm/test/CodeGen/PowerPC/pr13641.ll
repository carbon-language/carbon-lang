; RUN: llc -verify-machineinstrs < %s | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define void @foo() nounwind {
  ret void
}

; CHECK: blr
; CHECK-NEXT: .long 0
; CHECK-NEXT: .quad 0
