; RUN: llvm-upgrade < %s | llvm-as | llc

long %test(long %A) {
	%B = cast long %A to sbyte
	%C = cast sbyte %B to long
	ret long %C
}
