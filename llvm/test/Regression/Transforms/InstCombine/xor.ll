; This test makes sure that these instructions are properly eliminated.
;

; RUN: as < %s | opt -instcombine | dis | not grep 'xor '

implementation

bool %test0(bool %A) {
	%B = xor bool %A, false
	ret bool %B
}

int %test1(int %A) {
	%B = xor int %A, 0
	ret int %B
}

bool %test2(bool %A) {
	%B = xor bool %A, %A
	ret bool %B
}

int %test3(int %A) {
	%B = xor int %A, %A
	ret int %B
}

int %test4(int %A) {    ; A ^ ~A == -1
        %NotA = xor int -1, %A
        %B = xor int %A, %NotA
        ret int %B
}

uint %test5(uint %A) { ; (A|B)^B == A & (~B)
	%t1 = or uint %A, 123
	%r  = xor uint %t1, 123
	ret uint %r
}

ubyte %test6(ubyte %A) {
	%B = xor ubyte %A, 17
	%C = xor ubyte %B, 17
	ret ubyte %C
}

; (A & C1)^(B & C2) -> (A & C1)|(B & C2) iff C1&C2 == 0
int %test7(int %A, int %B) {

        %A1 = and int %A, 7
        %B1 = and int %B, 128
        %C1 = xor int %A1, %B1
        ret int %C1
}

ubyte %test8(bool %c) {
	%d = xor bool %c, true    ; invert the condition
	br bool %d, label %True, label %False
True:
	ret ubyte 1
False:
	ret ubyte 3
}

bool %test9(ubyte %A) {
	%B = xor ubyte %A, 123      ; xor can be eliminated
	%C = seteq ubyte %B, 34
	ret bool %C
}

ubyte %test10(ubyte %A) {
	%B = and ubyte %A, 3
	%C = xor ubyte %B, 4        ; transform into an OR
	ret ubyte %C
}

ubyte %test11(ubyte %A) {
	%B = or ubyte %A, 12
	%C = xor ubyte %B, 4        ; transform into an AND
	ret ubyte %C
}

bool %test12(ubyte %A) {
	%B = xor ubyte %A, 4
	%c = setne ubyte %B, 0
	ret bool %c
}
