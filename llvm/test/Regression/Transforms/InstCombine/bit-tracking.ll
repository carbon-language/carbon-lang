; This file contains various testcases that require tracking whether bits are
; set or cleared by various instructions.

; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep %ELIM

; test1 - Eliminating the casts in this testcase (by narrowing the AND 
; operation) allows instcombine to realize the function always returns false.
;
bool %test1(int %A, int %B) {
        %C1 = setlt int %A, %B
        %ELIM1 = cast bool %C1 to uint
        %C2 = setgt int %A, %B
        %ELIM2 = cast bool %C2 to uint
        %C3 = and uint %ELIM1, %ELIM2
        %ELIM3 = cast uint %C3 to bool
        ret bool %ELIM3
}

; See if we can eliminate the shifts...
int %test2(int %B) {
        %ELIM1 = shl int %B, ubyte 31
        %ELIM2 = shr int %ELIM1, ubyte 31
        %inc = add int %ELIM2, 1   ; == xor int %B, 1
        ret int %inc
}

; Reduce down to a single XOR
int %test3(int %B) {
        %ELIMinc = and int %B, 1
        %tmp.5 = xor int %ELIMinc, 1
        %ELIM7 = and int %B, -2
        %tmp.8 = or int %tmp.5, %ELIM7
        ret int %tmp.8
}

; Finally, a bigger case where we chain things together.  This corresponds to
; incrementing a single-bit bitfield, which should become just an xor.
int %test4(int %B) {
        %ELIM3 = shl int %B, ubyte 31
        %ELIM4 = shr int %ELIM3, ubyte 31
        %inc = add int %ELIM4, 1
        %ELIM5 = and int %inc, 1
        %ELIM7 = and int %B, -2
        %tmp.8 = or int %ELIM5, %ELIM7
        ret int %tmp.8
}

