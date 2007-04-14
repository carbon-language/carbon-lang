; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | \
; RUN:    grep and | wc -l | grep 1

; Should be optimized to one and.
bool %test1(uint %a, uint %b) {
        %tmp1 = and uint %a, 65280
        %tmp3 = and uint %b, 65280
        %tmp = setne uint %tmp1, %tmp3
        ret bool %tmp
}

