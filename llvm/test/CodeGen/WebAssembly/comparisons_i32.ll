; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that basic 32-bit integer comparison operations assemble as expected.

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: eq_i32:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: @1{{$}}
; CHECK-NEXT: set_local @2, pop{{$}}
; CHECK-NEXT: @0{{$}}
; CHECK-NEXT: set_local @3, pop{{$}}
; CHECK-NEXT: eq @3, @2{{$}}
; CHECK-NEXT: set_local @4, pop{{$}}
; CHECK-NEXT: return @4{{$}}
define i32 @eq_i32(i32 %x, i32 %y) {
  %a = icmp eq i32 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ne_i32:
; CHECK: ne @3, @2{{$}}
; CHECK-NEXT: set_local @4, pop{{$}}
define i32 @ne_i32(i32 %x, i32 %y) {
  %a = icmp ne i32 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: slt_i32:
; CHECK: slt @3, @2{{$}}
; CHECK-NEXT: set_local @4, pop{{$}}
define i32 @slt_i32(i32 %x, i32 %y) {
  %a = icmp slt i32 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: sle_i32:
; CHECK: sle @3, @2{{$}}
; CHECK-NEXT: set_local @4, pop{{$}}
define i32 @sle_i32(i32 %x, i32 %y) {
  %a = icmp sle i32 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ult_i32:
; CHECK: ult @3, @2{{$}}
; CHECK-NEXT: set_local @4, pop{{$}}
define i32 @ult_i32(i32 %x, i32 %y) {
  %a = icmp ult i32 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ule_i32:
; CHECK: ule @3, @2{{$}}
; CHECK-NEXT: set_local @4, pop{{$}}
define i32 @ule_i32(i32 %x, i32 %y) {
  %a = icmp ule i32 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: sgt_i32:
; CHECK: sgt @3, @2{{$}}
; CHECK-NEXT: set_local @4, pop{{$}}
define i32 @sgt_i32(i32 %x, i32 %y) {
  %a = icmp sgt i32 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: sge_i32:
; CHECK: sge @3, @2{{$}}
; CHECK-NEXT: set_local @4, pop{{$}}
define i32 @sge_i32(i32 %x, i32 %y) {
  %a = icmp sge i32 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ugt_i32:
; CHECK: ugt @3, @2{{$}}
; CHECK-NEXT: set_local @4, pop{{$}}
define i32 @ugt_i32(i32 %x, i32 %y) {
  %a = icmp ugt i32 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: uge_i32:
; CHECK: uge @3, @2{{$}}
; CHECK-NEXT: set_local @4, pop{{$}}
define i32 @uge_i32(i32 %x, i32 %y) {
  %a = icmp uge i32 %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}
