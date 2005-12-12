; RUN: llvm-as < %s | llc

long %test(long %A) {
	%B = cast long %A to sbyte
	%C = cast sbyte %B to long
	ret long %C
}
