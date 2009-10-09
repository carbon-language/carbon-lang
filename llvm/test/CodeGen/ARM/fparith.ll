; RUN: llc < %s -march=arm -mattr=+vfp2 | FileCheck %s

define float @f1(float %a, float %b) {
;CHECK: f1:
;CHECK: fadds
entry:
	%tmp = fadd float %a, %b		; <float> [#uses=1]
	ret float %tmp
}

define double @f2(double %a, double %b) {
;CHECK: f2:
;CHECK: faddd
entry:
	%tmp = fadd double %a, %b		; <double> [#uses=1]
	ret double %tmp
}

define float @f3(float %a, float %b) {
;CHECK: f3:
;CHECK: fmuls
entry:
	%tmp = fmul float %a, %b		; <float> [#uses=1]
	ret float %tmp
}

define double @f4(double %a, double %b) {
;CHECK: f4:
;CHECK: fmuld
entry:
	%tmp = fmul double %a, %b		; <double> [#uses=1]
	ret double %tmp
}

define float @f5(float %a, float %b) {
;CHECK: f5:
;CHECK: fsubs
entry:
	%tmp = fsub float %a, %b		; <float> [#uses=1]
	ret float %tmp
}

define double @f6(double %a, double %b) {
;CHECK: f6:
;CHECK: fsubd
entry:
	%tmp = fsub double %a, %b		; <double> [#uses=1]
	ret double %tmp
}

define float @f7(float %a) {
;CHECK: f7:
;CHECK: eor
entry:
	%tmp1 = fsub float -0.000000e+00, %a		; <float> [#uses=1]
	ret float %tmp1
}

define double @f8(double %a) {
;CHECK: f8:
;CHECK: fnegd
entry:
	%tmp1 = fsub double -0.000000e+00, %a		; <double> [#uses=1]
	ret double %tmp1
}

define float @f9(float %a, float %b) {
;CHECK: f9:
;CHECK: fdivs
entry:
	%tmp1 = fdiv float %a, %b		; <float> [#uses=1]
	ret float %tmp1
}

define double @f10(double %a, double %b) {
;CHECK: f10:
;CHECK: fdivd
entry:
	%tmp1 = fdiv double %a, %b		; <double> [#uses=1]
	ret double %tmp1
}

define float @f11(float %a) {
;CHECK: f11:
;CHECK: bic
entry:
	%tmp1 = call float @fabsf( float %a )		; <float> [#uses=1]
	ret float %tmp1
}

declare float @fabsf(float)

define double @f12(double %a) {
;CHECK: f12:
;CHECK: fabsd
entry:
	%tmp1 = call double @fabs( double %a )		; <double> [#uses=1]
	ret double %tmp1
}

declare double @fabs(double)
