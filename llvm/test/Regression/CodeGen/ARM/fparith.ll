; RUN: llvm-as < %s | llc -march=arm &&
; RUN: llvm-as < %s | llc -march=arm | grep fadds &&
; RUN: llvm-as < %s | llc -march=arm | grep faddd &&
; RUN: llvm-as < %s | llc -march=arm | grep fmuls &&
; RUN: llvm-as < %s | llc -march=arm | grep fmuld &&
; RUN: llvm-as < %s | llc -march=arm | grep fnegs &&
; RUN: llvm-as < %s | llc -march=arm | grep fnegd &&
; RUN: llvm-as < %s | llc -march=arm | grep fdivs &&
; RUN: llvm-as < %s | llc -march=arm | grep fdivd

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

float %f7(float %a) {
entry:
	%tmp1 = sub float -0.000000e+00, %a
	ret float %tmp1
}

double %f8(double %a) {
entry:
	%tmp1 = sub double -0.000000e+00, %a
	ret double %tmp1
}

float %f9(float %a, float %b) {
entry:
	%tmp1 = div float %a, %b
	ret float %tmp1
}

double %f10(double %a, double %b) {
entry:
	%tmp1 = div double %a, %b
	ret double %tmp1
}
