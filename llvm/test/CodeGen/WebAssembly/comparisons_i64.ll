; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that basic 64-bit integer comparison operations assemble as expected.

target datalayout = "e-p:32:32-i64:64-v128:8:128-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: eq_i64:
; CHECK-NEXT: (setlocal @0 (argument 1))
; CHECK-NEXT: (setlocal @1 (argument 0))
; CHECK-NEXT: (setlocal @2 (eq @1 @0))
; CHECK-NEXT: (return @2)
define i32 @eq_i64(i64 %x, i64 %y) {
  %a = icmp eq i64 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ne_i64:
; CHECK: (setlocal @2 (ne @1 @0))
define i32 @ne_i64(i64 %x, i64 %y) {
  %a = icmp ne i64 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: slt_i64:
; CHECK: (setlocal @2 (slt @1 @0))
define i32 @slt_i64(i64 %x, i64 %y) {
  %a = icmp slt i64 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: sle_i64:
; CHECK: (setlocal @2 (sle @1 @0))
define i32 @sle_i64(i64 %x, i64 %y) {
  %a = icmp sle i64 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ult_i64:
; CHECK: (setlocal @2 (ult @1 @0))
define i32 @ult_i64(i64 %x, i64 %y) {
  %a = icmp ult i64 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ule_i64:
; CHECK: (setlocal @2 (ule @1 @0))
define i32 @ule_i64(i64 %x, i64 %y) {
  %a = icmp ule i64 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: sgt_i64:
; CHECK: (setlocal @2 (sgt @1 @0))
define i32 @sgt_i64(i64 %x, i64 %y) {
  %a = icmp sgt i64 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: sge_i64:
; CHECK: (setlocal @2 (sge @1 @0))
define i32 @sge_i64(i64 %x, i64 %y) {
  %a = icmp sge i64 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ugt_i64:
; CHECK: (setlocal @2 (ugt @1 @0))
define i32 @ugt_i64(i64 %x, i64 %y) {
  %a = icmp ugt i64 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: uge_i64:
; CHECK: (setlocal @2 (uge @1 @0))
define i32 @uge_i64(i64 %x, i64 %y) {
  %a = icmp uge i64 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}
