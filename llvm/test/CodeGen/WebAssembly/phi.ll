; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that phis are lowered.

target datalayout = "e-p:32:32-i64:64-v128:8:128-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: test0
; CHECK: (setlocal [[REG:@.*]] (argument 0))
; CHECK: (setlocal [[REG]] (sdiv [[REG]] {{.*}}))
; CHECK: (return [[REG]])
define i32 @test0(i32 %p) {
entry:
  %t = icmp slt i32 %p, 0
  br i1 %t, label %true, label %done
true:
  %a = sdiv i32 %p, 3
  br label %done
done:
  %s = phi i32 [ %a, %true ], [ %p, %entry ]
  ret i32 %s
}
