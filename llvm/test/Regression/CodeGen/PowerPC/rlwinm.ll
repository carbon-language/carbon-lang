; All of these ands and shifts should be folded into rlwimi's
; RUN: llvm-as < rlwinm.ll | llc -march=ppc32 | not grep and && 
; RUN: llvm-as < rlwinm.ll | llc -march=ppc32 | grep rlwinm | wc -l | grep 2

implementation   ; Functions:

int %test1(int %a) {
entry:
        %tmp.1 = and int %a, 268431360          ; <int> [#uses=1]
        ret int %tmp.1
}

int %test2(int %a) {
entry:
        %tmp.1 = and int %a, -268435441         ; <int> [#uses=1]
        ret int %tmp.1
}
