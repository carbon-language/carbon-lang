; This test makes sure that these instructions are properly eliminated.
;

; RUN: as < %s | opt -instcombine -die | dis | grep-not phi

implementation

int %test1(int %A, bool %b) {
BB0:    br bool %b, label %BB1, label %BB2
BB1:
        %B = phi int [%A, %BB0]     ; Combine away one argument PHI nodes
        ret int %B
BB2:
        ret int %A
}

int %test2(int %A, bool %b) {
BB0:    br bool %b, label %BB1, label %BB2
BB1:
	br label %BB2
BB2:
        %B = phi int [%A, %BB0], [%A, %BB1]     ; Combine away PHI nodes with same values
        ret int %B
}

int %test3(int %A, bool %b) {
BB0: br label %Loop

Loop:
	%B = phi int [%A, %BB0], [%B, %Loop]    ; PHI has same value always.
	br bool %b, label %Loop, label %Exit
Exit:
	ret int %B
}

int %test3(bool %b) {
BB0:  ret int 7                                 ; Loop is unreachable

Loop:
        %B = phi int [%B, %L2], [%B, %Loop]     ; PHI has same value always.
        br bool %b, label %L2, label %Loop
L2:
	br label %Loop
}

