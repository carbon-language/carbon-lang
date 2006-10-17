; RUN: llvm-as < %s | llc -march=arm &&
; RUN: llvm-as < %s | llc -march=arm | grep fmsr &&
; RUN: llvm-as < %s | llc -march=arm | grep fmrs &&
; RUN: llvm-as < %s | llc -march=arm | grep fmrrd &&
; RUN: llvm-as < %s | llc -march=arm | grep fmdrr &&
; RUN: llvm-as < %s | llc -march=arm | grep fldd &&
; RUN: llvm-as < %s | llc -march=arm | grep flds &&
; RUN: llvm-as < %s | llc -march=arm | grep fstd &&
; RUN: llvm-as < %s | llc -march=arm | grep fsts &&
; RUN: llvm-as < %s | llc -march=arm | grep ".word.*1065353216"


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
