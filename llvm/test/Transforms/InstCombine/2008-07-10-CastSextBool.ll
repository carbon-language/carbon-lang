; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {%C = xor i1 %A, true}
; PR2539

define i1 @test(i1 %A) {
	%B = zext i1 %A to i32
	%C = icmp slt i32 %B, 1
	ret i1 %C
}
