; RUN: llvm-as < %s | llc -march=arm &&
; RUN: llvm-as < %s | llc -march=arm | grep fcvtds &&
; RUN: llvm-as < %s | llc -march=arm | grep fcvtsd

float %f(double %x) {
entry:
	%tmp1 = cast double %x to float
	ret float %tmp1
}

double %g(float %x) {
entry:
	%tmp1 = cast float %x to double
	ret double %tmp1
}
