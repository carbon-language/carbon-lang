; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   llc -march=ppc32 -mtriple=powerpc-apple-darwin8 -mattr=stfiwx -o %t1 -f
; RUN: grep stfiwx %t1
; RUN: not grep r1 %t1
; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   llc -march=ppc32 -mtriple=powerpc-apple-darwin8 -mattr=-stfiwx \
; RUN:   -o %t2 -f
; RUN: not grep stfiwx %t2
; RUN: grep r1 %t2

void %test(float %a, int* %b) {
        %tmp.2 = cast float %a to int
        store int %tmp.2, int* %b
        ret void
}

void %test2(float %a, int* %b, int %i) {
        %tmp.2 = getelementptr int* %b, int 1
        %tmp.5 = getelementptr int* %b, int %i
        %tmp.7 = cast float %a to int
        store int %tmp.7, int* %tmp.5
        store int %tmp.7, int* %tmp.2
        store int %tmp.7, int* %b
        ret void
}

