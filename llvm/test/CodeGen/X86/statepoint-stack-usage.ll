; RUN: llc < %s | FileCheck %s

target datalayout = "e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; This test is checking to make sure that we reuse the same stack slots
; for GC values spilled over two different call sites.  Since the order
; of GC arguments differ, niave lowering code would insert loads and 
; stores to rearrange items on the stack.  We need to make sure (for
; performance) that this doesn't happen.
define i32 @back_to_back_calls(i32* %a, i32* %b, i32* %c) #1 {
; CHECK-LABEL: back_to_back_calls
; The exact stores don't matter, but there need to be three stack slots created
; CHECK: movq	%rdx, 16(%rsp)
; CHECK: movq	%rdi, 8(%rsp)
; CHECK: movq	%rsi, (%rsp)
  %safepoint_token = tail call i32 (void ()*, i32, i32, ...)* @llvm.experimental.gc.statepoint.p0f_isVoidf(void ()* undef, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0, i32* %a, i32* %b, i32* %c)
  %a1 = tail call coldcc i32* @llvm.experimental.gc.relocate.p0i32(i32 %safepoint_token, i32 9, i32 9)
  %b1 = tail call coldcc i32* @llvm.experimental.gc.relocate.p0i32(i32 %safepoint_token, i32 9, i32 10)
  %c1 = tail call coldcc i32* @llvm.experimental.gc.relocate.p0i32(i32 %safepoint_token, i32 9, i32 11)
; CHECK: callq
; This is the key check.  There should NOT be any memory moves here
; CHECK-NOT: movq
  %safepoint_token2 = tail call i32 (void ()*, i32, i32, ...)* @llvm.experimental.gc.statepoint.p0f_isVoidf(void ()* undef, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0, i32* %c1, i32* %b1, i32* %a1)
  %a2 = tail call coldcc i32* @llvm.experimental.gc.relocate.p0i32(i32 %safepoint_token2, i32 9, i32 11)
  %b2 = tail call coldcc i32* @llvm.experimental.gc.relocate.p0i32(i32 %safepoint_token2, i32 9, i32 10)
  %c2 = tail call coldcc i32* @llvm.experimental.gc.relocate.p0i32(i32 %safepoint_token2, i32 9, i32 9)
; CHECK: callq
  ret i32 1
}

; This test simply checks that minor changes in vm state don't prevent slots
; being reused for gc values.  
define i32 @reserve_first(i32* %a, i32* %b, i32* %c) #1 {
; CHECK-LABEL: reserve_first
; The exact stores don't matter, but there need to be three stack slots created
; CHECK: movq	%rdx, 16(%rsp)
; CHECK: movq	%rdi, 8(%rsp)
; CHECK: movq	%rsi, (%rsp)
  %safepoint_token = tail call i32 (void ()*, i32, i32, ...)* @llvm.experimental.gc.statepoint.p0f_isVoidf(void ()* undef, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0, i32* %a, i32* %b, i32* %c)
  %a1 = tail call coldcc i32* @llvm.experimental.gc.relocate.p0i32(i32 %safepoint_token, i32 9, i32 9)
  %b1 = tail call coldcc i32* @llvm.experimental.gc.relocate.p0i32(i32 %safepoint_token, i32 9, i32 10)
  %c1 = tail call coldcc i32* @llvm.experimental.gc.relocate.p0i32(i32 %safepoint_token, i32 9, i32 11)
; CHECK: callq
; This is the key check.  There should NOT be any memory moves here
; CHECK-NOT: movq
  %safepoint_token2 = tail call i32 (void ()*, i32, i32, ...)* @llvm.experimental.gc.statepoint.p0f_isVoidf(void ()* undef, i32 0, i32 0, i32 5, i32* %a1, i32 0, i32* %c1, i32 0, i32 0, i32* %c1, i32* %b1, i32* %a1)
  %a2 = tail call coldcc i32* @llvm.experimental.gc.relocate.p0i32(i32 %safepoint_token2, i32 9, i32 11)
  %b2 = tail call coldcc i32* @llvm.experimental.gc.relocate.p0i32(i32 %safepoint_token2, i32 9, i32 10)
  %c2 = tail call coldcc i32* @llvm.experimental.gc.relocate.p0i32(i32 %safepoint_token2, i32 9, i32 9)
; CHECK: callq
  ret i32 1
}

; Function Attrs: nounwind
declare i32* @llvm.experimental.gc.relocate.p0i32(i32, i32, i32) #3

declare i32 @llvm.experimental.gc.statepoint.p0f_isVoidf(void ()*, i32, i32, ...)

attributes #1 = { uwtable }
