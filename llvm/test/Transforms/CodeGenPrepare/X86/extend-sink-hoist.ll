; RUN: opt -codegenprepare -disable-cgp-branch-opts -S < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; The first cast should be sunk into block2, in order that the
; instruction selector can form an efficient
; i64 * i64 -> i128 multiplication.
define i128 @sink(i64* %mem1, i64* %mem2) {
; CHECK-LABEL: block1:
; CHECK-NEXT: load
block1:
  %l1 = load i64* %mem1
  %s1 = sext i64 %l1 to i128
  br label %block2

; CHECK-LABEL: block2:
; CHECK-NEXT: sext
; CHECK-NEXT: load
; CHECK-NEXT: sext
block2:
  %l2 = load i64* %mem2
  %s2 = sext i64 %l2 to i128
  %res = mul i128 %s1, %s2
  ret i128 %res
}

; The first cast should be hoisted into block1, in order that the
; instruction selector can form an extend-load.
define i64 @hoist(i32* %mem1, i32* %mem2) {
; CHECK-LABEL: block1:
; CHECK-NEXT: load
; CHECK-NEXT: sext
block1:
  %l1 = load i32* %mem1
  br label %block2

; CHECK-LABEL: block2:
; CHECK-NEXT: load
; CHECK-NEXT: sext
block2:
  %s1 = sext i32 %l1 to i64
  %l2 = load i32* %mem2
  %s2 = sext i32 %l2 to i64
  %res = mul i64 %s1, %s2
  ret i64 %res
}
