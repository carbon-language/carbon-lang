; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86 | grep and | wc -l | grep 1

; The dag combiner should fold together (x&127)|(y&16711680) -> (x|y)&c1
; in this case.
uint %test6(uint %x, ushort %y) {
        %tmp1 = cast ushort %y to uint
        %tmp2 = and uint %tmp1, 127             ; <uint> [#uses=1]
        %tmp4 = shl uint %x, ubyte 16           ; <uint> [#uses=1]
        %tmp5 = and uint %tmp4, 16711680                ; <uint> [#uses=1]
        %tmp6 = or uint %tmp2, %tmp5            ; <uint> [#uses=1]
        ret uint %tmp6
}

