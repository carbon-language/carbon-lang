; RUN: llvm-as < %s | opt -fdle | llvm-dis | grep NOTDEAD

define void @test(i32* %Q, i32* %P) {
  %A = load i32* %Q
	%NOTDEAD = volatile load i32* %Q		; <i32> [#uses=1]
	ret void
}
