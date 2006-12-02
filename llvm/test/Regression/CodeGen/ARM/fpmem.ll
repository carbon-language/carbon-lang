; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | grep flds | wc -l | grep 2 &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | grep "flds.*\[" | wc -l | grep 1 &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | grep "fsts.*\[" | wc -l | grep 1

float %f1(float %a) {
entry:
	ret float 0.000000e+00
}

float %f2(float* %v) {
entry:
        %tmp = load float* %v
	ret float %tmp
}

void %f3(float %a, float* %v) {
entry:
	store float %a, float* %v
	ret void
}
