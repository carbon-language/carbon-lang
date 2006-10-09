; RUN: llvm-as < %s | llc -march=arm &&
; RUN: llvm-as < %s | llc -march=arm | grep fadds &&
; RUN: llvm-as < %s | llc -march=arm | grep faddd &&
; RUN: llvm-as < %s | llc -march=arm | grep fmuls &&
; RUN: llvm-as < %s | llc -march=arm | grep fmuld

float %f(float %a, float %b) {
entry:
	%tmp = add float %a, %b
	ret float %tmp
}

double %g(double %a, double %b) {
entry:
	%tmp = add double %a, %b
	ret double %tmp
}

float %h(float %a, float %b) {
entry:
	%tmp = mul float %a, %b
	ret float %tmp
}

double %i(double %a, double %b) {
entry:
	%tmp = mul double %a, %b
	ret double %tmp
}
