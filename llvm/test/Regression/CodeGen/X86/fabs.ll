; Make sure this testcase codegens to the fabs instruction, not a call to fabsf
; RUN: llvm-as < %s | llc -march=x86 | grep 'fabs$' | wc -l | grep 2

declare float %fabsf(float)

float %fabsftest(float %X) {
        %Y = call float %fabsf(float %X)
        ret float %Y
}

double %fabstest2(double %X) {
        %Y = setge double %X, -0.0
        %Z = sub double -0.0, %X
        %Q = select bool %Y, double %X, double %Z
        ret double %Q
}

