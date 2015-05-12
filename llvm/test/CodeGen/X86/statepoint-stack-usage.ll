; RUN: llc < %s | FileCheck %s

target datalayout = "e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; This test is checking to make sure that we reuse the same stack slots
; for GC values spilled over two different call sites.  Since the order
; of GC arguments differ, niave lowering code would insert loads and 
; stores to rearrange items on the stack.  We need to make sure (for
; performance) that this doesn't happen.
define i32 @back_to_back_calls(i32 addrspace(1)* %a, i32 addrspace(1)* %b, i32 addrspace(1)* %c) #1 gc "statepoint-example" {
; CHECK-LABEL: back_to_back_calls
; The exact stores don't matter, but there need to be three stack slots created
; CHECK: movq	%rdi, 16(%rsp)
; CHECK: movq	%rdx, 8(%rsp)
; CHECK: movq	%rsi, (%rsp)
  %safepoint_token = tail call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* undef, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0, i32 addrspace(1)* %a, i32 addrspace(1)* %b, i32 addrspace(1)* %c)
  %a1 = tail call coldcc i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(i32 %safepoint_token, i32 12, i32 12)
  %b1 = tail call coldcc i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(i32 %safepoint_token, i32 12, i32 13)
  %c1 = tail call coldcc i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(i32 %safepoint_token, i32 12, i32 14)
; CHECK: callq
; This is the key check.  There should NOT be any memory moves here
; CHECK-NOT: movq
  %safepoint_token2 = tail call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* undef, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0, i32 addrspace(1)* %c1, i32 addrspace(1)* %b1, i32 addrspace(1)* %a1)
  %a2 = tail call coldcc i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(i32 %safepoint_token2, i32 12, i32 14)
  %b2 = tail call coldcc i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(i32 %safepoint_token2, i32 12, i32 13)
  %c2 = tail call coldcc i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(i32 %safepoint_token2, i32 12, i32 12)
; CHECK: callq
  ret i32 1
}

; This test simply checks that minor changes in vm state don't prevent slots
; being reused for gc values.  
define i32 @reserve_first(i32 addrspace(1)* %a, i32 addrspace(1)* %b, i32 addrspace(1)* %c) #1 gc "statepoint-example" {
; CHECK-LABEL: reserve_first
; The exact stores don't matter, but there need to be three stack slots created
; CHECK: movq	%rdi, 16(%rsp)
; CHECK: movq	%rdx, 8(%rsp)
; CHECK: movq	%rsi, (%rsp)
  %safepoint_token = tail call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* undef, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0, i32 addrspace(1)* %a, i32 addrspace(1)* %b, i32 addrspace(1)* %c)
  %a1 = tail call coldcc i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(i32 %safepoint_token, i32 12, i32 12)
  %b1 = tail call coldcc i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(i32 %safepoint_token, i32 12, i32 13)
  %c1 = tail call coldcc i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(i32 %safepoint_token, i32 12, i32 14)
; CHECK: callq
; This is the key check.  There should NOT be any memory moves here
; CHECK-NOT: movq
  %safepoint_token2 = tail call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* undef, i32 0, i32 0, i32 0, i32 5, i32 addrspace(1)* %a1, i32 0, i32 addrspace(1)* %c1, i32 0, i32 0, i32 addrspace(1)* %c1, i32 addrspace(1)* %b1, i32 addrspace(1)* %a1)
  %a2 = tail call coldcc i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(i32 %safepoint_token2, i32 12, i32 14)
  %b2 = tail call coldcc i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(i32 %safepoint_token2, i32 12, i32 13)
  %c2 = tail call coldcc i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(i32 %safepoint_token2, i32 12, i32 12)
; CHECK: callq
  ret i32 1
}

; Function Attrs: nounwind
declare i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(i32, i32, i32) #3

declare i32 @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)

attributes #1 = { uwtable }