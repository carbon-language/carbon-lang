; Dominance relationships is not calculated correctly for unreachable blocks,
; which causes the verifier to barf on this input.

int %test(bool %b) {
BB0:  ret int 7                                 ; Loop is unreachable

Loop:
        %B = phi int [%B, %L2], [%B, %Loop]     ; PHI has same value always.
        br bool %b, label %L2, label %Loop
L2:
        br label %Loop
}

