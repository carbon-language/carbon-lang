; This test makes sure that add instructions are properly eliminated.

; RUN: llvm-as < %s | opt -instcombine -die | llvm-dis | grep -v OK | not grep add

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

int %test9(int %A) {
	%B = shl int %A, ubyte 4
	%C = add int %B, %B      ; === shl int %A, 5
	ret int %C
}

bool %test10(ubyte %A, ubyte %b) {
        %B = add ubyte %A, %b
        %c = setne ubyte %B, 0    ; === A != -b
        ret bool %c
}

bool %test11(ubyte %A) {
        %B = add ubyte %A, 255
        %c = setne ubyte %B, 0    ; === A != 1
        ret bool %c
}

int %test12(int %A, int %B) {
	%C_OK = add int %B, %A       ; Should be transformed into shl A, 1
	br label %X
X:
	%D = add int %C_OK, %A 
	ret int %D
}

int %test13(int %A, int %B, int %C) {
	%D_OK = add int %A, %B
	%E_OK = add int %D_OK, %C
	%F = add int %E_OK, %A        ;; shl A, 1
	ret int %F
}

uint %test14(uint %offset, uint %difference) {
        %tmp.2 = and uint %difference, 3
        %tmp.3_OK = add uint %tmp.2, %offset
        %tmp.5.mask = and uint %difference, 4294967292
        %tmp.8 = add uint %tmp.3_OK, %tmp.5.mask ; == add %offset, %difference
        ret uint %tmp.8
}

