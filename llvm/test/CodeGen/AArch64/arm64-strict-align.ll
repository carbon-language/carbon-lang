; RUN: llc < %s -mtriple=arm64-apple-darwin | FileCheck %s
; RUN: llc < %s -mtriple=arm64-apple-darwin -aarch64-no-strict-align | FileCheck %s
; RUN: llc < %s -mtriple=arm64-apple-darwin -aarch64-strict-align | FileCheck %s --check-prefix=CHECK-STRICT
; RUN: llc < %s -mtriple=arm64-apple-darwin -aarch64-strict-align -fast-isel | FileCheck %s --check-prefix=CHECK-STRICT

define i32 @f0(i32* nocapture %p) nounwind {
; CHECK-STRICT: ldrh [[HIGH:w[0-9]+]], [x0, #2]
; CHECK-STRICT: ldrh [[LOW:w[0-9]+]], [x0]
; CHECK-STRICT: bfi [[LOW]], [[HIGH]], #16, #16
; CHECK-STRICT: ret

; CHECK: ldr w0, [x0]
; CHECK: ret
  %tmp = load i32, i32* %p, align 2
  ret i32 %tmp
}

define i64 @f1(i64* nocapture %p) nounwind {
; CHECK-STRICT:	ldp	w[[LOW:[0-9]+]], w[[HIGH:[0-9]+]], [x0]
; CHECK-STRICT: bfi x[[LOW]], x[[HIGH]], #32, #32
; CHECK-STRICT:	ret

; CHECK: ldr x0, [x0]
; CHECK: ret
  %tmp = load i64, i64* %p, align 4
  ret i64 %tmp
}
