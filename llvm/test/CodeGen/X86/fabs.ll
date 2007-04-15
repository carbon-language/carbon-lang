; Make sure this testcase codegens to the fabs instruction, not a call to fabsf
; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86 -mattr=-sse2,-sse3 | \
; RUN:   grep fabs\$ | wc -l | grep 1
; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   llc -march=x86 -mattr=-sse2,-sse3 -enable-unsafe-fp-math  | \
; RUN:   grep fabs\$ | wc -l | grep 2

target endian = little
target pointersize = 32

declare float %fabsf(float)

float %test1(float %X) {
        %Y = call float %fabsf(float %X)
        ret float %Y
}

double %test2(double %X) {
        %Y = setge double %X, -0.0
        %Z = sub double -0.0, %X
        %Q = select bool %Y, double %X, double %Z
        ret double %Q
}

