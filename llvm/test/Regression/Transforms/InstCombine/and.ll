; This test makes sure that these instructions are properly eliminated.
;

; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep and

implementation

int %test1(int %A) {
	%B = and int %A, 0     ; zero result
	ret int %B
}

int %test2(int %A) {
	%B = and int %A, -1    ; noop
	ret int %B
}

bool %test3(bool %A) {
	%B = and bool %A, false  ; always = false
	ret bool %B
}

bool %test4(bool %A) {
	%B = and bool %A, true  ; noop
	ret bool %B
}

int %test5(int %A) {
	%B = and int %A, %A
	ret int %B
}

bool %test6(bool %A) {
	%B = and bool %A, %A
	ret bool %B
}

int %test7(int %A) {         ; A & ~A == 0
        %NotA = xor int %A, -1
        %B = and int %A, %NotA
        ret int %B
}

ubyte %test8(ubyte %A) {    ; AND associates
	%B = and ubyte %A, 3
	%C = and ubyte %B, 4
	ret ubyte %C
}

bool %test9(int %A) {
	%B = and int %A, -2147483648   ; Test of sign bit, convert to setle %A, 0 
	%C = setne int %B, 0
	ret bool %C
}

bool %test9(uint %A) {
	%B = and uint %A, 2147483648   ; Test of sign bit, convert to setle %A, 0 
	%C = setne uint %B, 0
	ret bool %C
}

uint %test10(uint %A) {
	%B = and uint %A, 12
	%C = xor uint %B, 15
	%D = and uint %C, 1   ; (X ^ C1) & C2 --> (X & C2) ^ (C1&C2)
	ret uint %D
}

uint %test11(uint %A, uint* %P) {
	%B = or uint %A, 3
	%C = xor uint %B, 12
	store uint %C, uint* %P    ; additional use of C
	%D = and uint %C, 3        ; %C = and uint %B, 3 --> 3
	ret uint %D
}

bool %test12(uint %A, uint %B) {
	%C1 = setlt uint %A, %B
	%C2 = setle uint %A, %B
	%D = and bool %C1, %C2      ; (A < B) & (A <= B) === (A < B)
	ret bool %D
}

bool %test13(uint %A, uint %B) {
	%C1 = setlt uint %A, %B
	%C2 = setgt uint %A, %B
	%D = and bool %C1, %C2      ; (A < B) & (A > B) === false
	ret bool %D
}

bool %test14(ubyte %A) {
	%B = and ubyte %A, 128
	%C = setne ubyte %B, 0
	ret bool %C
}

ubyte %test15(ubyte %A) {
	%B = shr ubyte %A, ubyte 7
	%C = and ubyte %B, 2        ; Always equals zero
	ret ubyte %C
}

ubyte %test16(ubyte %A) {
	%B = shl ubyte %A, ubyte 2
	%C = and ubyte %B, 3
	ret ubyte %C
}

sbyte %test17(sbyte %X, sbyte %Y) { ;; ~(~X & Y) --> (X | ~Y)
	%B = xor sbyte %X, -1
	%C = and sbyte %B, %Y
        %D = xor sbyte %C, -1
	ret sbyte %D
}

