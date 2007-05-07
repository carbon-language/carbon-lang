; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+vfp2 > %t
; RUN: grep fmsr %t | wc -l | grep 4
; RUN: grep fsitos %t
; RUN: grep fmrs %t | wc -l | grep 2
; RUN: grep fsitod %t
; RUN: grep fmrrd %t | wc -l | grep 5
; RUN: grep fmdrr %t | wc -l | grep 2
; RUN: grep fldd %t
; RUN: grep fuitod %t
; RUN: grep fuitos %t
; RUN: grep 1065353216 %t

float %f(int %a) {
entry:
	%tmp = cast int %a to float		; <float> [#uses=1]
	ret float %tmp
}

double %g(int %a) {
entry:
        %tmp = cast int %a to double            ; <double> [#uses=1]
        ret double %tmp
}

double %uint_to_double(uint %a) {
entry:
	%tmp = cast uint %a to double
	ret double %tmp
}

float %uint_to_float(uint %a) {
entry:
	%tmp = cast uint %a to float
	ret float %tmp
}


double %h(double* %v) {
entry:
	%tmp = load double* %v		; <double> [#uses=1]
	ret double %tmp
}

float %h2() {
entry:
        ret float 1.000000e+00
}

double %f2(double %a) {
        ret double %a
}

void %f3() {
entry:
	%tmp = call double %f5()		; <double> [#uses=1]
	call void %f4(double %tmp )
	ret void
}

declare void %f4(double)
declare double %f5()
