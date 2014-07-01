; RUN: opt < %s -globalopt -S | FileCheck %s

global i32 0
; CHECK: @0 = internal global i32 0
define i32* @1() {
	ret i32* @0
}
; CHECK: define internal fastcc i32* @1()
define i32* @f() {
entry:
	call i32* @1()
	ret i32* %0
}
