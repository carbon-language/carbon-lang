; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine -disable-output &&
; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | not grep undef

int %test(sbyte %A) {
	%B = cast sbyte %A to int
	%C = shr int %B, ubyte 8
	ret int %C
}

