; RUN: llc %s -o - -fast-isel=true -O1 -verify-machineinstrs | FileCheck %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios8.0.0"

; The machine verifier was asserting on this test because the AND instruction was
; sunk below the test which killed %tmp340.
; The kill flags on the test had to be cleared because the AND was going to read
; registers in a BB after the test instruction.

; CHECK: %bb343
; CHECK: and

define i32 @test(i32* %ptr) {
bb:
  br label %.thread

.thread:                                          ; preds = %.thread, %bb
  %loc = phi i32 [ %next_iter, %.thread ], [ 0, %bb ]
  %next_iter = lshr i32 %loc, 1
  %tmp340 = sub i32 %loc, 1
  %tmp341 = and i32 %tmp340, 1
  %tmp342 = icmp eq i32 %tmp341, 0
  br i1 %tmp342, label %bb343, label %.thread

bb343:                                            ; preds = %.thread
  store i32 %tmp341, i32* %ptr, align 4
  ret i32 -1
}
