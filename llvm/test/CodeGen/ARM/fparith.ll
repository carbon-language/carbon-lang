; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+vfp2 &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+vfp2 | grep fadds &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+vfp2 | grep faddd &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+vfp2 | grep fmuls &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+vfp2 | grep fmuld &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+vfp2 | grep fnegs &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+vfp2 | grep fnegd &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+vfp2 | grep fdivs &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+vfp2 | grep fdivd


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

float %f11(float %a) {
entry:
	%tmp1 = call float %fabsf(float %a)
	ret float %tmp1
}

declare float %fabsf(float)

double %f12(double %a) {
entry:
	%tmp1 = call double %fabs(double %a)
	ret double %tmp1
}

declare double %fabs(double)
