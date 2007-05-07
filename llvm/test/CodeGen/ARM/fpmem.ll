; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | \
; RUN:   grep {mov r0, #0} | wc -l | grep 1
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+vfp2 | \
; RUN:   grep {flds.*\\\[} | wc -l | grep 1
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+vfp2 | \
; RUN:   grep {fsts.*\\\[} | wc -l | grep 1

float %f1(float %a) {
	ret float 0.000000e+00
}

float %f2(float* %v, float %u) {
        %tmp = load float* %v
        %tmp1 = add float %tmp, %u
	ret float %tmp1
}

void %f3(float %a, float %b, float* %v) {
        %tmp = add float %a, %b
	store float %tmp, float* %v
	ret void
}
