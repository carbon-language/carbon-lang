; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -stats |& \
; RUN:   grep {Number of machine instrs printed} | grep 12

ushort %Trans16Bit(uint %srcA, uint %srcB, uint %alpha) {
        %tmp1 = shl uint %srcA, ubyte 15                ; <uint> [#uses=1]
        %tmp2 = and uint %tmp1, 32505856                ; <uint> [#uses=1]
        %tmp4 = and uint %srcA, 31775           ; <uint> [#uses=1]
        %tmp5 = or uint %tmp2, %tmp4            ; <uint> [#uses=1]
        %tmp7 = shl uint %srcB, ubyte 15                ; <uint> [#uses=1]
        %tmp8 = and uint %tmp7, 32505856                ; <uint> [#uses=1]
        %tmp10 = and uint %srcB, 31775          ; <uint> [#uses=1]
        %tmp11 = or uint %tmp8, %tmp10          ; <uint> [#uses=1]
        %tmp14 = mul uint %tmp5, %alpha         ; <uint> [#uses=1]
        %tmp16 = sub uint 32, %alpha            ; <uint> [#uses=1]
        %tmp18 = mul uint %tmp11, %tmp16                ; <uint> [#uses=1]
        %tmp19 = add uint %tmp18, %tmp14                ; <uint> [#uses=2]
        %tmp21 = shr uint %tmp19, ubyte 5               ; <uint> [#uses=1]
        %tmp21 = cast uint %tmp21 to ushort             ; <ushort> [#uses=1]
        %tmp = and ushort %tmp21, 31775         ; <ushort> [#uses=1]
        %tmp23 = shr uint %tmp19, ubyte 20              ; <uint> [#uses=1]
        %tmp23 = cast uint %tmp23 to ushort             ; <ushort> [#uses=1]
        %tmp24 = and ushort %tmp23, 992         ; <ushort> [#uses=1]
        %tmp25 = or ushort %tmp, %tmp24         ; <ushort> [#uses=1]
        ret ushort %tmp25
}

