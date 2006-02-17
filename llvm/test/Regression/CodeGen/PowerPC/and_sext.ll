; This test should not contain a sign extend
; RUN: llvm-as < %s | llc -march=ppc32 | not grep extsh 

int %test(uint %mode.0.i.0) {
        %tmp.79 = cast uint %mode.0.i.0 to short        ; <sbyte> [#uses=1]
        %tmp.80 = cast short %tmp.79 to int             ; <int> [#uses=1]
        %tmp.81 = and int %tmp.80, 24             ; <int> [#uses=1]
        ret int %tmp.81
}
