; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | \
; RUN:   grep {mov r0, #0} | wc -l | grep 1
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+vfp2 | \
; RUN:   grep {flds.*\\\[} | wc -l | grep 1
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+vfp2 | \
; RUN:   grep {fsts.*\\\[} | wc -l | grep 1

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
