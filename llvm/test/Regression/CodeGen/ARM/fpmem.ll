; RUN: llvm-as < %s | llc -march=arm &&
; RUN: llvm-as < %s | llc -march=arm | grep flds | wc -l | grep 2 &&
; RUN: llvm-as < %s | llc -march=arm | grep "flds.*\[" | wc -l | grep 1

float %g(float %a) {
entry:
	ret float 0.000000e+00
}

float %g(float* %v) {
entry:
        %tmp = load float* %v
	ret float %tmp
}
