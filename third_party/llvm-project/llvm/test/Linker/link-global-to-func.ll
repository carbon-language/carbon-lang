; RUN: llvm-as %s -o %t1.bc
; RUN: echo "declare void @__eprintf(i8*, i8*, i32, i8*) noreturn     define void @foo() {      tail call void @__eprintf( i8* undef, i8* undef, i32 4, i8* null ) noreturn nounwind       unreachable }" | llvm-as -o %t2.bc
; RUN: llvm-link %t2.bc %t1.bc -S | FileCheck %s
; RUN: llvm-link %t1.bc %t2.bc -S | FileCheck %s
; CHECK: __eprintf

; rdar://6072702

@__eprintf = external global i8*		; <i8**> [#uses=1]

define i8* @test() {
	%A = load i8*, i8** @__eprintf		; <i8*> [#uses=1]
	ret i8* %A
}
