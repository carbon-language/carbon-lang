; test that malloc's with a constant argument are promoted to array allocations
; RUN: opt < %s -instcombine -S | grep getelementptr

define i32* @test() {
	%X = malloc i32, i32 4
	ret i32* %X
}
