; RUN: llvm-as < %s | llc -march=ppc32 | grep fmul | wc -l | grep 2
; RUN: llvm-as < %s | llc -march=ppc32 -enable-unsafe-fp-math | \
; RUN:   grep fmul | wc -l | grep 1

define double @foo(double %X) {
        %tmp1 = mul double %X, 1.23
        %tmp2 = mul double %tmp1, 4.124
        ret double %tmp2
}

