; RUN: llvm-as < %s

; This testcase was previously considered invalid for indexing into a pointer
; that is contained WITHIN a structure, but this is now valid.

define void @test({i32, i32*} * %X) {
	getelementptr {i32, i32*} * %X, i32 0, i32 1, i32 0
	ret void
}
