; RUN: llvm-as < %s | llc -march=arm &&
; RUN: llvm-as < %s | llc -march=arm | grep fmsr  | wc -l | grep 5 &&
; RUN: llvm-as < %s | llc -march=arm | grep fsitos &&
; RUN: llvm-as < %s | llc -march=arm | grep fmrs &&
; RUN: llvm-as < %s | llc -march=arm | grep fsitod &&
; RUN: llvm-as < %s | llc -march=arm | grep fmrrd | wc -l | grep 5 &&
; RUN: llvm-as < %s | llc -march=arm | grep fmdrr | wc -l | grep 3 &&
; RUN: llvm-as < %s | llc -march=arm | grep fldd &&
; RUN: llvm-as < %s | llc -march=arm | grep flds &&
; RUN: llvm-as < %s | llc -march=arm | grep fstd &&
; RUN: llvm-as < %s | llc -march=arm | grep fsts &&
; RUN: llvm-as < %s | llc -march=arm | grep fuitod &&
; RUN: llvm-as < %s | llc -march=arm | grep fuitos &&
; RUN: llvm-as < %s | llc -march=arm | grep ".word.*1065353216"

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

float %h() {
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

void %f6(float %a, float* %b) {
entry:
	store float %a, float* %b
	ret void
}

void %f7(double %a, double* %b) {
entry:
	store double %a, double* %b
	ret void
}
