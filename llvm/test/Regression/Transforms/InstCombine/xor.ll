; This test makes sure that these instructions are properly eliminated.
;

; RUN: as < %s | opt -instcombine | dis | not grep 'xor '

implementation

bool %test5(bool %A) {
	%B = xor bool %A, false
	ret bool %B
}

int %test6(int %A) {
	%B = xor int %A, 0
	ret int %B
}

bool %test7(bool %A) {
	%B = xor bool %A, %A
	ret bool %B
}

int %test8(int %A) {
	%B = xor int %A, %A
	ret int %B
}

int %test11(int %A) {    ; A ^ ~A == -1
        %NotA = xor int -1, %A
        %B = xor int %A, %NotA
        ret int %B
}

uint %test13(uint %A) { ; (A|B)^B == A & (~B)
	%t1 = or uint %A, 123
	%r  = xor uint %t1, 123
	ret uint %r
}

ubyte %test15(ubyte %A) {
	%B = xor ubyte %A, 17
	%C = xor ubyte %B, 17
	ret ubyte %C
}

int %test16(int %A, int %B) {     ; (A & C1)^(B & C2) -> (A & C1)|(B & C2) iff C1&C2 == 0
        %A1 = and int %A, 7
        %B1 = and int %B, 128
        %OROK = xor int %A1, %B1
        ret int %OROK
}

ubyte %test18(bool %c) {
	%d = xor bool %c, true    ; invert the condition
	br bool %d, label %True, label %False
True:
	ret ubyte 1
False:
	ret ubyte 3
}

bool %test19(ubyte %A) {
	%B = xor ubyte %A, 123      ; xor can be eliminated
	%C = seteq ubyte %B, 34
	ret bool %C
}
