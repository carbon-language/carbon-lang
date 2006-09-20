%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Fix cmovs with a constant on the wrong side

aka:
        lda $0,10($31)
        cmovlt $17,$0,$16

is bad for:

long %cmov_lt2(long %a, long %c) {
entry:
	%tmp.1 = setlt long %c, 0
	%retval = select bool %tmp.1, long 10, long %a
	ret long %retval
}

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

