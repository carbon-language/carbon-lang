; RUN: llvm-as < %s | llc -march=arm &&
; RUN: llvm-as < %s | llc -march=arm | grep fadds &&
; RUN: llvm-as < %s | llc -march=arm | grep faddd &&
; RUN: llvm-as < %s | llc -march=arm | grep fmuls &&
; RUN: llvm-as < %s | llc -march=arm | grep fmuld

float %f1(float %a, float %b) {
entry:
	%tmp = add float %a, %b
	ret float %tmp
}

double %f2(double %a, double %b) {
entry:
	%tmp = add double %a, %b
	ret double %tmp
}

float %f3(float %a, float %b) {
entry:
	%tmp = mul float %a, %b
	ret float %tmp
}

double %f4(double %a, double %b) {
entry:
	%tmp = mul double %a, %b
	ret double %tmp
}

float %f5(float %a, float %b) {
entry:
	%tmp = sub float %a, %b
	ret float %tmp
}

double %f6(double %a, double %b) {
entry:
	%tmp = sub double %a, %b
	ret double %tmp
}
