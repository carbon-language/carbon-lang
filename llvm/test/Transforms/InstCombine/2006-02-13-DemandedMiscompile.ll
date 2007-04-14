; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | \
; RUN:   not grep undef

int %test(sbyte %A) {
	%B = cast sbyte %A to int
	%C = shr int %B, ubyte 8
	ret int %C
}

