; RUN: llc < %s | FileCheck %s
; This file contains a collection of basic tests to ensure we didn't
; screw up normal call lowering when there are no deopt or gc arguments.

target datalayout = "e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

declare zeroext i1 @return_i1()
declare zeroext i32 @return_i32()
declare i32* @return_i32ptr()
declare float @return_float()

define i1 @test_i1_return() {
; CHECK-LABEL: test_i1_return
; This is just checking that a i1 gets lowered normally when there's no extra
; state arguments to the statepoint
; CHECK: pushq %rax
; CHECK: callq return_i1
; CHECK: popq %rdx
; CHECK: retq
entry:
  %safepoint_token = tail call i32 (i1 ()*, i32, i32, ...)* @llvm.experimental.gc.statepoint.p0f_i1f(i1 ()* @return_i1, i32 0, i32 0, i32 0)
  %call1 = call zeroext i1 @llvm.experimental.gc.result.int.i1(i32 %safepoint_token)
  ret i1 %call1
}

define i32 @test_i32_return() {
; CHECK-LABEL: test_i32_return
; CHECK: pushq %rax
; CHECK: callq return_i32
; CHECK: popq %rdx
; CHECK: retq
entry:
  %safepoint_token = tail call i32 (i32 ()*, i32, i32, ...)* @llvm.experimental.gc.statepoint.p0f_i32f(i32 ()* @return_i32, i32 0, i32 0, i32 0)
  %call1 = call zeroext i32 @llvm.experimental.gc.result.int.i32(i32 %safepoint_token)
  ret i32 %call1
}

define i32* @test_i32ptr_return() {
; CHECK-LABEL: test_i32ptr_return
; CHECK: pushq %rax
; CHECK: callq return_i32ptr
; CHECK: popq %rdx
; CHECK: retq
entry:
  %safepoint_token = tail call i32 (i32* ()*, i32, i32, ...)* @llvm.experimental.gc.statepoint.p0f_p0i32f(i32* ()* @return_i32ptr, i32 0, i32 0, i32 0)
  %call1 = call i32* @llvm.experimental.gc.result.ptr.p0i32(i32 %safepoint_token)
  ret i32* %call1
}

define float @test_float_return() {
; CHECK-LABEL: test_float_return
; CHECK: pushq %rax
; CHECK: callq return_float
; CHECK: popq %rax
; CHECK: retq
entry:
  %safepoint_token = tail call i32 (float ()*, i32, i32, ...)* @llvm.experimental.gc.statepoint.p0f_f32f(float ()* @return_float, i32 0, i32 0, i32 0)
  %call1 = call float @llvm.experimental.gc.result.float.f32(i32 %safepoint_token)
  ret float %call1
}

declare i32 @llvm.experimental.gc.statepoint.p0f_i1f(i1 ()*, i32, i32, ...)
declare i1 @llvm.experimental.gc.result.int.i1(i32)

declare i32 @llvm.experimental.gc.statepoint.p0f_i32f(i32 ()*, i32, i32, ...)
declare i32 @llvm.experimental.gc.result.int.i32(i32)

declare i32 @llvm.experimental.gc.statepoint.p0f_p0i32f(i32* ()*, i32, i32, ...)
declare i32* @llvm.experimental.gc.result.ptr.p0i32(i32)

declare i32 @llvm.experimental.gc.statepoint.p0f_f32f(float ()*, i32, i32, ...)
declare float @llvm.experimental.gc.result.float.f32(i32)

