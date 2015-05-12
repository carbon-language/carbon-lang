; RUN: not opt -verify 2>&1 < %s | FileCheck %s

; CHECK: gc.statepoint must have null as call target if number of patchable bytes is non zero

define i1 @invalid_patchable_statepoint() gc "statepoint-example" {
entry:
  %safepoint_token = tail call i32 (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 0, i32 3, i1 ()* @func, i32 0, i32 0, i32 0, i32 0)
  %call1 = call i1 @llvm.experimental.gc.result.i1(i32 %safepoint_token)
  ret i1 %call1
}

declare i32 @llvm.experimental.gc.statepoint.p0f_i1f(i64, i32, i1 ()*, i32, i32, ...)
declare i1 @llvm.experimental.gc.result.i1(i32)
declare i1 @func()
