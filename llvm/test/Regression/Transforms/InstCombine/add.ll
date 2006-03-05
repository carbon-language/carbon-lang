; This test makes sure that add instructions are properly eliminated.

; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep -v OK | not grep add

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

ubyte %test15(ubyte %A) {
        %B = add ubyte %A, 192  ; Does not effect result
        %C = and ubyte %B, 16   ; Only one bit set
        ret ubyte %C
}

ubyte %test16(ubyte %A) {
        %B = add ubyte %A, 16   ; Turn this into a XOR
        %C = and ubyte %B, 16   ; Only one bit set
        ret ubyte %C
}

int %test17(int %A) {
        %B = xor int %A, -1
        %C = add int %B, 1      ; == sub int 0, %A
        ret int %C
}

ubyte %test18(ubyte %A) {
        %B = xor ubyte %A, 255
        %C = add ubyte %B, 17      ; == sub ubyte 16, %A
        ret ubyte %C
}

int %test19(bool %C) {
        %A = select bool %C, int 1000, int 10
        %V = add int %A, 123
        ret int %V
}

int %test20(int %x) {
        %tmp.2 = xor int %x, -2147483648
        ;; Add of sign bit -> xor of sign bit.
        %tmp.4 = add int %tmp.2, -2147483648
        ret int %tmp.4
}

bool %test21(uint %x) {
	%t = add uint %x, 4
	%y = seteq uint %t, 123
	ret bool %y
}

int %test22(uint %V) {
	%V2 = add uint %V, 10
	switch uint %V2, label %Default [
		uint 20, label %Lab1
		uint 30, label %Lab2
	]
Default:
	ret int 123
Lab1:
	ret int 12312
Lab2:
	ret int 1231231
}

int %test23(bool %C, int %a) {
entry:
        br bool %C, label %endif, label %else

else:
        br label %endif

endif:
        %b.0 = phi int [ 0, %entry ], [ 1, %else ]
        %tmp.4 = add int %b.0, 1
        ret int %tmp.4
}

int %test24(int %A) {
	%B = add int %A, 1
	%C = shl int %B, ubyte 1
	%D = sub int %C, 2
	ret int %D             ;; A << 1
}

long %test25(long %Y) {
        %tmp.4 = shl long %Y, ubyte 2
        %tmp.12 = shl long %Y, ubyte 2
        %tmp.8 = add long %tmp.4, %tmp.12 ;; Y << 3
        ret long %tmp.8
}

int %test26(int %A, int %B) {
	%C = add int %A, %B
	%D = sub int %C, %B
	ret int %D
}

int %test27(bool %C, int %X, int %Y) {
        %A = add int %X, %Y
        %B = add int %Y, 123
        %C = select bool %C, int %A, int %B  ;; Fold add through select.
        %D = sub int %C, %Y
        ret int %D
}

int %test28(int %X) {
	%Y = add int %X, 1234
	%Z = sub int 42, %Y
	ret int %Z
}

uint %test29(uint %X, uint %x) {
	%tmp.2 = sub uint %X, %x
        %tmp.2.mask = and uint %tmp.2, 63               ; <uint> [#uses=1]
        %tmp.6 = add uint %tmp.2.mask, %x               ; <uint> [#uses=1]
        %tmp.7 = and uint %tmp.6, 63            ; <uint> [#uses=1]
        %tmp.9 = and uint %tmp.2, 4294967232            ; <uint> [#uses=1]
        %tmp.10 = or uint %tmp.7, %tmp.9                ; <uint> [#uses=1]
	ret uint %tmp.10
}

long %test30(long %x) {
        %tmp.2 = xor long %x, -9223372036854775808
        ;; Add of sign bit -> xor of sign bit.
        %tmp.4 = add long %tmp.2, -9223372036854775808
        ret long %tmp.4
}

int %test31(int %A) {
	%B = add int %A, 4
	%C = mul int %B, 5
	%D = sub int %C, 20
	ret int %D
}

int %test32(int %A) {
	%B = add int %A, 4
	%C = shl int %B, ubyte 2
	%D = sub int %C, 16
	ret int %D
}
