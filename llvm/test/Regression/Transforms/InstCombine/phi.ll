; This test makes sure that these instructions are properly eliminated.
;

; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep phi

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

int %test4(bool %b) {
BB0:  ret int 7                                 ; Loop is unreachable

Loop:
        %B = phi int [%B, %L2], [%B, %Loop]     ; PHI has same value always.
        br bool %b, label %L2, label %Loop
L2:
	br label %Loop
}

int %test5(int %A, bool %b) {
BB0: br label %Loop

Loop:
        %B = phi int [%A, %BB0], [undef, %Loop]    ; PHI has same value always.
        br bool %b, label %Loop, label %Exit
Exit:
        ret int %B
}

uint %test6(int %A, bool %b) {
BB0:
        %X = cast int %A to uint
        br bool %b, label %BB1, label %BB2
BB1:
        %Y = cast int %A to uint
        br label %BB2
BB2:
        %B = phi uint [%X, %BB0], [%Y, %BB1] ;; Suck casts into phi
        ret uint %B
}

