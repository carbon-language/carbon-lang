; Make sure this testcase codegens to the fabs instruction, not a call to fabsf
; RUN: llvm-as < %s | llc -disable-pattern-isel=0 | grep 'fabs$'

declare float %fabsf(float)

float %fabsftest(float %X) {
        %Y = call float %fabsf(float %X)
        ret float %Y
}

