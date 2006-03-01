; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 &&
; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 | not grep r1

double %test1(double %X) {
        %Y = cast double %X to long
        %Z = cast long %Y to double
        ret double %Z
}

float %test2(double %X) {
        %Y = cast double %X to long
        %Z = cast long %Y to float
        ret float %Z
}

double %test3(float %X) {
        %Y = cast float %X to long
        %Z = cast long %Y to double
        ret double %Z
}

float %test4(float %X) {
        %Y = cast float %X to long
        %Z = cast long %Y to float
        ret float %Z
}

