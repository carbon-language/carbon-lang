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

bool %test18(int %A) {
	%B = and int %A, -128
	%C = setne int %B, 0   ;; C >= 128
	ret bool %C
}

bool %test18a(ubyte %A) {
	%B = and ubyte %A, 254
	%C = seteq ubyte %B, 0
	ret bool %C
}

int %test19(int %A) {
	%B = shl int %A, ubyte 3
	%C = and int %B, -2    ;; Clearing a zero bit
	ret int %C
}

ubyte %test20(ubyte %A) {
	%C = shr ubyte %A, ubyte 7 
	%D = and ubyte %C, 1            ;; Unneeded
	ret ubyte %D
}

bool %test22(int %A) {
	%B = seteq int %A, 1
	%C = setge int %A, 3
	%D = and bool %B, %C   ;; False
	ret bool %D
}

bool %test23(int %A) {
	%B = setgt int %A, 1
	%C = setle int %A, 2
	%D = and bool %B, %C   ;; A == 2
	ret bool %D
}

bool %test24(int %A) {
	%B = setgt int %A, 1
	%C = setne int %A, 2
	%D = and bool %B, %C   ;; A > 2
	ret bool %D
}

bool %test25(int %A) {
	%B = setge int %A, 50
	%C = setlt int %A, 100
	%D = and bool %B, %C   ;; (A-50) <u 50
	ret bool %D
}

bool %test26(int %A) {
        %B = setne int %A, 50
        %C = setne int %A, 51
        %D = and bool %B, %C   ;; (A-50) > 1
        ret bool %D
}

ubyte %test27(ubyte %A) {
	%B = and ubyte %A, 4
	%C = sub ubyte %B, 16
	%D = and ubyte %C, 240   ;; 0xF0
	%E = add ubyte %D, 16
	ret ubyte %E
}

int %test28(int %X) {       ;; This is juse a zero extending shr.
        %Y = shr int %X, ubyte 24  ;; Sign extend
        %Z = and int %Y, 255       ;; Mask out sign bits
        ret int %Z
}

int %test29(ubyte %X) {
        %Y = cast ubyte %X to int
        %Z = and int %Y, 255       ;; Zero extend makes this unneeded.
        ret int %Z
}

int %test30(bool %X) {
	%Y = cast bool %X to int
	%Z = and int %Y, 1
	ret int %Z
}

uint %test31(bool %X) {
	%Y = cast bool %X to uint
	%Z = shl uint %Y, ubyte 4
	%A = and uint %Z, 16
	ret uint %A
}

uint %test32(uint %In) {
	%Y = and uint %In, 16
	%Z = shr uint %Y, ubyte 2
	%A = and uint %Z, 1
	ret uint %A
}

uint %test33(uint %b) {   ;; Code corresponding to one-bit bitfield ^1.
        %tmp.4.mask = and uint %b, 1
        %tmp.10 = xor uint %tmp.4.mask, 1
        %tmp.12 = and uint %b, 4294967294
        %tmp.13 = or uint %tmp.12, %tmp.10
        ret uint %tmp.13
}

