; RUN: llvm-as < %s | llc -march=ppc32 -mattr=stfiwx | grep stfiwx &&
; RUN: llvm-as < %s | llc -march=ppc32 -mattr=stfiwx | not grep r1 &&
; RUN: llvm-as < %s | llc -march=ppc32 -mattr=-stfiwx | not grep stfiwx &&
; RUN: llvm-as < %s | llc -march=ppc32 -mattr=-stfiwx | grep r1

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

