; RUN: llvm-as < %s | opt -instcombine -disable-output &&
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep undef

int %test(sbyte %A) {
	%B = cast sbyte %A to int
	%C = shr int %B, ubyte 8
	ret int %C
}

