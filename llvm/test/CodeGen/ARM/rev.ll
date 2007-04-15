; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+v6 | grep rev16
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+v6 | grep revsh

int %test1(uint %X) {
        %tmp1 = shr uint %X, ubyte 8            ; <uint> [#uses=1]
        %tmp1 = cast uint %tmp1 to int          ; <int> [#uses=2]
        %X15 = cast uint %X to int              ; <int> [#uses=1]
        %tmp4 = shl int %X15, ubyte 8           ; <int> [#uses=2]
        %tmp2 = and int %tmp1, 16711680         ; <int> [#uses=1]
        %tmp5 = and int %tmp4, -16777216                ; <int> [#uses=1]
        %tmp9 = and int %tmp1, 255              ; <int> [#uses=1]
        %tmp13 = and int %tmp4, 65280           ; <int> [#uses=1]
        %tmp6 = or int %tmp5, %tmp2             ; <int> [#uses=1]
        %tmp10 = or int %tmp6, %tmp13           ; <int> [#uses=1]
        %tmp14 = or int %tmp10, %tmp9           ; <int> [#uses=1]
        ret int %tmp14
}

int %test2(uint %X) {   ; revsh
        %tmp1 = shr uint %X, ubyte 8            ; <uint> [#uses=1]
        %tmp1 = cast uint %tmp1 to short                ; <short> [#uses=1]
        %tmp3 = cast uint %X to short           ; <short> [#uses=1]
        %tmp2 = and short %tmp1, 255            ; <short> [#uses=1]
        %tmp4 = shl short %tmp3, ubyte 8                ; <short> [#uses=1]
        %tmp5 = or short %tmp2, %tmp4           ; <short> [#uses=1]
        %tmp5 = cast short %tmp5 to int         ; <int> [#uses=1]
        ret int %tmp5
}

