; These tests should not contain a sign extend.
; RUN: llvm-as < %s | llc -march=ppc32 &&
; RUN: llvm-as < %s | llc -march=ppc32 | not grep extsh  &&
; RUN: llvm-as < %s | llc -march=ppc32 | not grep extsb

int %test1(uint %mode.0.i.0) {
        %tmp.79 = cast uint %mode.0.i.0 to short
        %tmp.80 = cast short %tmp.79 to int
        %tmp.81 = and int %tmp.80, 24
        ret int %tmp.81
}

short %test2(short %X, short %x) {
        %tmp = cast short %X to int
        %tmp1 = cast short %x to int
        %tmp2 = add int %tmp, %tmp1
        %tmp4 = shr int %tmp2, ubyte 1
        %tmp4 = cast int %tmp4 to short
        %tmp45 = cast short %tmp4 to int
        %retval = cast int %tmp45 to short
        ret short %retval
}

