; RUN: llvm-as < %s | opt -load-vn -gcse | llvm-dis | not grep load
; Test that loads of undefined memory are eliminated.

define i32 @test1() {
	%X = malloc i32		; <i32*> [#uses=1]
	%Y = load i32* %X		; <i32> [#uses=1]
	ret i32 %Y
}

define i32 @test2() {
	%X = alloca i32		; <i32*> [#uses=1]
	%Y = load i32* %X		; <i32> [#uses=1]
	ret i32 %Y
}
