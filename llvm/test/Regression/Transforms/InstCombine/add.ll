; This test makes sure that add instructions are properly eliminated.
;
; This also tests that a subtract with a constant is properly converted
; to a add w/negative constant

; RUN: as < %s | opt -instcombine -die | dis | grep-not add

implementation

int %test1(int %A) {
	%B = add int %A, 0
	ret int %B
}

int %test2(int %A) {
	%B = add int %A, 5
	%C = add int %B, -5
	ret int %C
}

int %test3(int %A) {
	%B = add int %A, 5
	%C = sub int %B, 5   ;; This should get converted to an add
	ret int %C
}

int %test4(int %A, int %B) {
        %C = sub int 0, %A
        %D = add int %B, %C      ; D = B + -A = B - A
        ret int %D
}

int %test5(int %A, int %B) {
        %C = sub int 0, %A
        %D = add int %C, %B      ; D = -A + B = B - A
        ret int %D
}

int %test6(int %A) {
        %B = mul int 7, %A
        %C = add int %B, %A      ; C = 7*A+A == 8*A == A << 3
        ret int %C
}

int %test7(int %A) {
        %B = mul int 7, %A
        %C = add int %A, %B      ; C = A+7*A == 8*A == A << 3
        ret int %C
}

int %test8(int %A, int %B) {     ; (A & C1)+(B & C2) -> (A & C1)|(B & C2) iff C1&C2 == 0
	%A1 = and int %A, 7
	%B1 = and int %B, 128
	%C = add int %A1, %B1
	ret int %C
}

