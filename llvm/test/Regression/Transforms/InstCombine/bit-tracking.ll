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

