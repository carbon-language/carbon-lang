; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | \
; RUN:    grep {call.*llvm.bswap} | wc -l | grep 5
; END.

uint %test1(uint %i) {
        %tmp1 = shr uint %i, ubyte 24           ; <uint> [#uses=1]
        %tmp3 = shr uint %i, ubyte 8            ; <uint> [#uses=1]
        %tmp4 = and uint %tmp3, 65280           ; <uint> [#uses=1]
        %tmp5 = or uint %tmp1, %tmp4            ; <uint> [#uses=1]
        %tmp7 = shl uint %i, ubyte 8            ; <uint> [#uses=1]
        %tmp8 = and uint %tmp7, 16711680                ; <uint> [#uses=1]
        %tmp9 = or uint %tmp5, %tmp8            ; <uint> [#uses=1]
        %tmp11 = shl uint %i, ubyte 24          ; <uint> [#uses=1]
        %tmp12 = or uint %tmp9, %tmp11          ; <uint> [#uses=1]
        ret uint %tmp12
}

uint %test2(uint %arg) {
        %tmp2 = shl uint %arg, ubyte 24         ; <uint> [#uses=1]
        %tmp4 = shl uint %arg, ubyte 8          ; <uint> [#uses=1]
        %tmp5 = and uint %tmp4, 16711680                ; <uint> [#uses=1]
        %tmp6 = or uint %tmp2, %tmp5            ; <uint> [#uses=1]
        %tmp8 = shr uint %arg, ubyte 8          ; <uint> [#uses=1]
        %tmp9 = and uint %tmp8, 65280           ; <uint> [#uses=1]
        %tmp10 = or uint %tmp6, %tmp9           ; <uint> [#uses=1]
        %tmp12 = shr uint %arg, ubyte 24                ; <uint> [#uses=1]
        %tmp14 = or uint %tmp10, %tmp12         ; <uint> [#uses=1]
        ret uint %tmp14
}

ushort %test3(ushort %s) {
        %tmp2 = shr ushort %s, ubyte 8
        %tmp4 = shl ushort %s, ubyte 8
        %tmp5 = or ushort %tmp2, %tmp4
	ret ushort %tmp5
}

ushort %test4(ushort %s) {
        %tmp2 = shr ushort %s, ubyte 8
        %tmp4 = shl ushort %s, ubyte 8
        %tmp5 = or ushort %tmp4, %tmp2
	ret ushort %tmp5
}

; unsigned short test5(unsigned short a) {
;       return ((a & 0xff00) >> 8 | (a & 0x00ff) << 8);
;}
ushort %test5(ushort %a) {
        %tmp = zext ushort %a to int
        %tmp1 = and int %tmp, 65280
        %tmp2 = ashr int %tmp1, ubyte 8
        %tmp2 = trunc int %tmp2 to short
        %tmp4 = and int %tmp, 255
        %tmp5 = shl int %tmp4, ubyte 8
        %tmp5 = trunc int %tmp5 to short
        %tmp = or short %tmp2, %tmp5
        %tmp6 = bitcast short %tmp to ushort
        %tmp6 = zext ushort %tmp6 to int
        %retval = trunc int %tmp6 to ushort
        ret ushort %retval
}

