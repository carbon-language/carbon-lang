; This file contains various testcases that check to see that instcombine
; is narrowing computations when possible.
; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | \
; RUN:    grep {ret i1 false}

; test1 - Eliminating the casts in this testcase (by narrowing the AND
; operation) allows instcombine to realize the function always returns false.
;
bool %test1(int %A, int %B) {
        %C1 = setlt int %A, %B
        %ELIM1 = zext bool %C1 to uint
        %C2 = setgt int %A, %B
        %ELIM2 = zext bool %C2 to uint
        %C3 = and uint %ELIM1, %ELIM2
        %ELIM3 = trunc uint %C3 to bool
        ret bool %ELIM3
}
