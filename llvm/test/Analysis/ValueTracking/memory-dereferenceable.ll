; RUN: opt -print-memderefs -analyze -S <%s | FileCheck %s

; Uses the print-deref (+ analyze to print) pass to run
; isDereferenceablePointer() on many load instruction operands

target datalayout = "e"

declare zeroext i1 @return_i1()

@globalstr = global [6 x i8] c"hello\00"

define void @test(i32 addrspace(1)* dereferenceable(8) %dparam) {
; CHECK: The following are dereferenceable:
; CHECK: %globalptr
; CHECK: %alloca
; CHECK: %dparam
; CHECK: %relocate
; CHECK-NOT: %nparam
entry:
    %globalptr = getelementptr inbounds [6 x i8], [6 x i8]* @globalstr, i32 0, i32 0
    %load1 = load i8* %globalptr
    %alloca = alloca i1
    %load2 = load i1* %alloca
    %load3 = load i32 addrspace(1)* %dparam
    %tok = tail call i32 (i1 ()*, i32, i32, ...)* @llvm.experimental.gc.statepoint.p0f_i1f(i1 ()* @return_i1, i32 0, i32 0, i32 0, i32 addrspace(1)* %dparam)
    %relocate = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(i32 %tok, i32 4, i32 4)
    %load4 = load i32 addrspace(1)* %relocate
    %nparam = getelementptr i32, i32 addrspace(1)* %dparam, i32 5
    %load5 = load i32 addrspace(1)* %nparam
    ret void
}

declare i32 @llvm.experimental.gc.statepoint.p0f_i1f(i1 ()*, i32, i32, ...)
declare i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(i32, i32, i32)
