; This test makes sure that these instructions are properly eliminated.
;

; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep -v xor | not grep 'or '

implementation

int %test1(int %A) {
	%B = or int %A, 0
	ret int %B
}

int %test2(int %A) {
	%B = or int %A, -1
	ret int %B
}

ubyte %test2a(ubyte %A) {
	%B = or ubyte %A, 255
	ret ubyte %B
}

bool %test3(bool %A) {
	%B = or bool %A, false
	ret bool %B
}

bool %test4(bool %A) {
	%B = or bool %A, true
	ret bool %B
}

bool %test5(bool %A) {
	%B = or bool %A, %A
	ret bool %B
}

int %test6(int %A) {
	%B = or int %A, %A
	ret int %B
}

int %test7(int %A) {    ; A | ~A == -1
        %NotA = xor int -1, %A
        %B = or int %A, %NotA
        ret int %B
}

ubyte %test8(ubyte %A) {
	%B = or ubyte %A, 254
	%C = or ubyte %B, 1
	ret ubyte %C
}

ubyte %test9(ubyte %A, ubyte %B) {  ; Test that (A|c1)|(B|c2) == (A|B)|(c1|c2)
	%C = or ubyte %A, 1
	%D = or ubyte %B, 254
	%E = or ubyte %C, %D
	ret ubyte %E
}

ubyte %test10(ubyte %A) {
	%B = or ubyte %A, 1
	%C = and ubyte %B, 254
	%D = or ubyte %C, 254  ; (X & C1) | C2 --> (X | C2) & (C1|C2)
	ret ubyte %D
}

ubyte %test11(ubyte %A) {
	%B = or ubyte %A, 254
	%C = xor ubyte %B, 13
	%D = or ubyte %C, 1    ; (X ^ C1) | C2 --> (X | C2) ^ (C1&~C2)
	%E = xor ubyte %D, 12
	ret ubyte %E
}

uint %test12(uint %A) {
	%B = or uint %A, 4     ; Should be eliminated
	%C = and uint %B, 8
	ret uint %C
}

uint %test13(uint %A) {
	%B = or uint %A, 12
	%C = and uint %B, 8    ; Always equal to 8
	ret uint %C 
}

bool %test14(uint %A, uint %B) {
	%C1 = setlt uint %A, %B
	%C2 = setgt uint %A, %B
	%D = or bool %C1, %C2      ; (A < B) | (A > B) === A != B
	ret bool %D
}

bool %test15(uint %A, uint %B) {
        %C1 = setlt uint %A, %B
        %C2 = seteq uint %A, %B
        %D = or bool %C1, %C2      ; (A < B) | (A == B) === A <= B
        ret bool %D
}

int %test16(int %A) {
	%B = and int %A, 1
	%C = and int %A, -2       ; -2 = ~1
	%D = or int %B, %C        ; %D = and int %B, -1 == %B
	ret int %D
}

int %test17(int %A) {
	%B = and int %A, 1
	%C = and int %A, 4
	%D = or int %B, %C        ; %D = and int %B, 5
	ret int %D
}
