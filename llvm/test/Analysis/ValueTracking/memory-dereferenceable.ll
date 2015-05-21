; RUN: opt -print-memderefs -analyze -S <%s | FileCheck %s

; Uses the print-deref (+ analyze to print) pass to run
; isDereferenceablePointer() on many load instruction operands

target datalayout = "e"

declare zeroext i1 @return_i1()

@globalstr = global [6 x i8] c"hello\00"
@globali32ptr = external global i32*

define void @test(i32 addrspace(1)* dereferenceable(8) %dparam) gc "statepoint-example" {
; CHECK: The following are dereferenceable:
; CHECK: %globalptr
; CHECK: %alloca
; CHECK: %dparam
; CHECK: %relocate
; CHECK-NOT: %nparam
; CHECK-NOT: %nd_load
; CHECK: %d4_load
; CHECK-NOT: %d2_load
; CHECK-NOT: %d_or_null_load
; CHECK: %d_or_null_non_null_load
entry:
    %globalptr = getelementptr inbounds [6 x i8], [6 x i8]* @globalstr, i32 0, i32 0
    %load1 = load i8, i8* %globalptr
    %alloca = alloca i1
    %load2 = load i1, i1* %alloca
    %load3 = load i32, i32 addrspace(1)* %dparam
    %tok = tail call i32 (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 0, i32 0, i1 ()* @return_i1, i32 0, i32 0, i32 0, i32 0, i32 addrspace(1)* %dparam)
    %relocate = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(i32 %tok, i32 7, i32 7)
    %load4 = load i32, i32 addrspace(1)* %relocate
    %nparam = getelementptr i32, i32 addrspace(1)* %dparam, i32 5
    %load5 = load i32, i32 addrspace(1)* %nparam

    ; Load from a non-dereferenceable load
    %nd_load = load i32*, i32** @globali32ptr
    %load6 = load i32, i32* %nd_load

    ; Load from a dereferenceable load
    %d4_load = load i32*, i32** @globali32ptr, !dereferenceable !0
    %load7 = load i32, i32* %d4_load

    ; Load from an offset not covered by the dereferenceable portion
    %d2_load = load i32*, i32** @globali32ptr, !dereferenceable !1
    %load8 = load i32, i32* %d2_load

    ; Load from a potentially null pointer with dereferenceable_or_null
    %d_or_null_load = load i32*, i32** @globali32ptr, !dereferenceable_or_null !0
    %load9 = load i32, i32* %d_or_null_load

    ; Load from a non-null pointer with dereferenceable_or_null
    %d_or_null_non_null_load = load i32*, i32** @globali32ptr, !nonnull !2, !dereferenceable_or_null !0
    %load10 = load i32, i32* %d_or_null_non_null_load

    ret void
}

declare i32 @llvm.experimental.gc.statepoint.p0f_i1f(i64, i32, i1 ()*, i32, i32, ...)
declare i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(i32, i32, i32)

!0 = !{i64 4}
!1 = !{i64 2}
!2 = !{}
