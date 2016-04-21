; RUN: opt -S < %s -instcombine | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.7.0"

; Check transforms involving atomic operations

define i32 @test1(i32* %p) {
; CHECK-LABEL: define i32 @test1(
; CHECK: %x = load atomic i32, i32* %p seq_cst, align 4
; CHECK: shl i32 %x, 1
  %x = load atomic i32, i32* %p seq_cst, align 4
  %y = load i32, i32* %p, align 4
  %z = add i32 %x, %y
  ret i32 %z
}

define i32 @test2(i32* %p) {
; CHECK-LABEL: define i32 @test2(
; CHECK: %x = load volatile i32, i32* %p, align 4
; CHECK: %y = load volatile i32, i32* %p, align 4
  %x = load volatile i32, i32* %p, align 4
  %y = load volatile i32, i32* %p, align 4
  %z = add i32 %x, %y
  ret i32 %z
}

; The exact semantics of mixing volatile and non-volatile on the same
; memory location are a bit unclear, but conservatively, we know we don't
; want to remove the volatile.
define i32 @test3(i32* %p) {
; CHECK-LABEL: define i32 @test3(
; CHECK: %x = load volatile i32, i32* %p, align 4
  %x = load volatile i32, i32* %p, align 4
  %y = load i32, i32* %p, align 4
  %z = add i32 %x, %y
  ret i32 %z
}

; FIXME: Forwarding from a stronger atomic is fine
define i32 @test4(i32* %p) {
; CHECK-LABEL: define i32 @test4(
; CHECK: %x = load atomic i32, i32* %p seq_cst, align 4
; CHECK: %y = load atomic i32, i32* %p unordered, align 4
  %x = load atomic i32, i32* %p seq_cst, align 4
  %y = load atomic i32, i32* %p unordered, align 4
  %z = add i32 %x, %y
  ret i32 %z
}

; Forwarding from a non-atomic is not.  (The earlier load 
; could in priciple be promoted to atomic and then forwarded, 
; but we can't just  drop the atomic from the load.)
define i32 @test5(i32* %p) {
; CHECK-LABEL: define i32 @test5(
; CHECK: %x = load atomic i32, i32* %p unordered, align 4
  %x = load atomic i32, i32* %p unordered, align 4
  %y = load i32, i32* %p, align 4
  %z = add i32 %x, %y
  ret i32 %z
}

