; RUN: opt < %s -instcombine -S | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

; Uses InstCombine with DataLayout to propagate dereferenceable
; attribute via gc.relocate: if the derived ptr is dereferenceable(N),
; then the return attribute of gc.relocate is dereferenceable(N).

declare zeroext i1 @return_i1()
declare i32 @llvm.experimental.gc.statepoint.p0f_i1f(i1 ()*, i32, i32, ...)
declare i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(i32, i32, i32)

define i32 addrspace(1)* @deref(i32 addrspace(1)* dereferenceable(8) %dparam) {
; Checks that a dereferenceabler pointer
; CHECK-LABEL: @deref
; CHECK: call dereferenceable(8)
entry:
    %load = load i32, i32 addrspace(1)* %dparam
    %tok = tail call i32 (i1 ()*, i32, i32, ...)* @llvm.experimental.gc.statepoint.p0f_i1f(i1 ()* @return_i1, i32 0, i32 0, i32 0, i32 addrspace(1)* %dparam)
    %relocate = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(i32 %tok, i32 4, i32 4)
    ret i32 addrspace(1)* %relocate
}