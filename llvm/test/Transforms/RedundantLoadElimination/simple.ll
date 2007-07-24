; RUN: llvm-as < %s | opt -rle | llvm-dis | not grep DEAD

define void @test(i32* %Q, i32* %P) {
  %A = load i32* %Q
	%DEAD = load i32* %Q		; <i32> [#uses=1]
	ret void
}
