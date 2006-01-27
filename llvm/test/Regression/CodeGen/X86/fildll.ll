; RUN: llvm-as < %s | llc -march=x86 -x86-asm-syntax=att -mattr=-sse2 | grep fildll | wc -l | grep 2

fastcc double %sint64_to_fp(long %X) {
	%R = cast long %X to double
	ret double %R
}

fastcc double %uint64_to_fp(ulong %X) {
	%R = cast ulong %X to double
	ret double %R
}
