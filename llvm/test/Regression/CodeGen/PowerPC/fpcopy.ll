; RUN: llvm-as < %s | llc -march=ppc32 &&
; RUN: llvm-as < %s | llc -march=ppc32 | not grep fmr

double %test(float %F) {
	%F = cast float %F to double
	ret double %F
}
