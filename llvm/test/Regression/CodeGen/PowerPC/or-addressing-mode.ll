; RUN: llvm-as < %s | llc &&
; RUN: llvm-as < %s | llc | not grep ori && 
; RUN: llvm-as < %s | llc | not grep rlwimi

int %test1(sbyte* %P) {  ;; or -> lwzx
        %tmp.2.i = cast sbyte* %P to uint
        %tmp.4.i = and uint %tmp.2.i, 4294901760
        %tmp.10.i = shr uint %tmp.2.i, ubyte 5
        %tmp.11.i = and uint %tmp.10.i, 2040
        %tmp.13.i = or uint %tmp.11.i, %tmp.4.i
        %tmp.14.i = cast uint %tmp.13.i to int*
        %tmp.3 = load int* %tmp.14.i
        ret int %tmp.3
}

int %test2(int %P) {    ;; or -> lwz
        %tmp.2 = shl int %P, ubyte 4
        %tmp.3 = or int %tmp.2, 2
        %tmp.4 = cast int %tmp.3 to int*
        %tmp.5 = load int* %tmp.4
        ret int %tmp.5
}

