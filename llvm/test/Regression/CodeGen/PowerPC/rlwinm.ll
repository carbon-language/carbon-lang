; All of these ands and shifts should be folded into rlwimi's
; RUN: llvm-as < %s | llc -march=ppc32 | not grep and && 
; RUN: llvm-as < %s | llc -march=ppc32 | not grep srawi && 
; RUN: llvm-as < %s | llc -march=ppc32 | not grep srwi && 
; RUN: llvm-as < %s | llc -march=ppc32 | not grep slwi && 
; RUN: llvm-as < %s | llc -march=ppc32 | grep rlwinm | wc -l | grep 8

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

int %test3(int %a) {
entry:
        %tmp.2 = shr int %a, ubyte 8            ; <int> [#uses=1]
        %tmp.3 = and int %tmp.2, 255            ; <int> [#uses=1]
        ret int %tmp.3
}

uint %test4(uint %a) {
entry:
        %tmp.3 = shr uint %a, ubyte 8           ; <uint> [#uses=1]
        %tmp.4 = and uint %tmp.3, 255           ; <uint> [#uses=1]
        ret uint %tmp.4
}

int %test5(int %a) {
entry:
        %tmp.2 = shl int %a, ubyte 8            ; <int> [#uses=1]
        %tmp.3 = and int %tmp.2, -8388608       ; <int> [#uses=1]
        ret int %tmp.3
}

int %test6(int %a) {
entry:
        %tmp.1 = and int %a, 65280               ; <int> [#uses=1]
        %tmp.2 = shr int %tmp.1, ubyte 8         ; <uint> [#uses=1]
        ret int %tmp.2
}

uint %test7(uint %a) {
entry:
        %tmp.1 = and uint %a, 65280              ; <uint> [#uses=1]
        %tmp.2 = shr uint %tmp.1, ubyte 8        ; <uint> [#uses=1]
        ret uint %tmp.2
}

int %test8(int %a) {
entry:
        %tmp.1 = and int %a, 16711680            ; <int> [#uses=1]
        %tmp.2 = shl int %tmp.1, ubyte 8         ; <int> [#uses=1]
        ret int %tmp.2
}

