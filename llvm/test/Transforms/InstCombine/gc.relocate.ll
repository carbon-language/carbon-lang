; RUN: opt < %s -datalayout -instcombine -S | FileCheck %s

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