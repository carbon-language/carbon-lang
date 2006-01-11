; RUN: llvm-as < %s | llc &&
; RUN: llvm-as < %s | llc | not grep ', f1'

target endian = big
target pointersize = 32
target triple = "powerpc-apple-darwin8.2.0"

; Dead argument should reserve an FP register.
double %bar(double %DEAD, double %X, double %Y) {
        %tmp.2 = add double %X, %Y
        ret double %tmp.2
}

