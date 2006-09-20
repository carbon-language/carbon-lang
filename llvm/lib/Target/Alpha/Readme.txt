%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Fix Ordered/Unordered FP stuff


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
improve bytezap opertunities
ulong %foo(ulong %y) {
entry:
        %tmp = and ulong %y,  65535
        %tmp2 = shr ulong %tmp,  ubyte 3
        ret ulong %tmp2
}


compiles to a 3 instruction sequence without instcombine
        zapnot $16,3,$0
        srl $0,3,$0
        ret $31,($26),1
 
After instcombine you get
ulong %foo(ulong %y) {
entry:
        %tmp = shr ulong %y, ubyte 3            ; <ulong> [#uses=1]
        %tmp2 = and ulong %tmp, 8191            ; <ulong> [#uses=1]
        ret ulong %tmp2
}

which compiles to
        lda $0,8191($31)
        srl $16,3,$1
        and $1,$0,$0
        ret $31,($26),1

