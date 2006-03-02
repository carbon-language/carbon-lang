; This test makes sure that these instructions are properly eliminated.
;
; RUN: llvm-as < %s | opt -instcombine -disable-output &&
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep rem

implementation

int %test1(int %A) {
	%B = rem int %A, 1    ; ISA constant 0
	ret int %B
}

int %test2(int %A) {          ; 0 % X = 0, we don't need to preserve traps
	%B = rem int 0, %A
	ret int %B
}

uint %test3(uint %A) {
	%B = rem uint %A, 8   ; & 7
	ret uint %B
}

bool %test3(int %A) {
	%B = rem int %A, -8   ; & 7
	%C = setne int %B, 0
	ret bool %C
}

uint %test4(uint %X, bool %C) {
	%V = select bool %C, uint 1, uint 8
	%R = rem uint %X, %V
	ret uint %R
}

uint %test5(uint %X, ubyte %B) {
        %Amt = shl uint 32, ubyte %B
        %V = rem uint %X, %Amt
        ret uint %V
}

int %test6(int %A) {
	%B = rem int %A, 0   ;; undef
	ret int %B
}

int %test7(int %A) {
	%B = mul int %A, 26
	%C = rem int %B, 13
	ret int %C
}

int %test8(int %A) {
	%B = shl int %A, ubyte 4
	%C = rem int %B, 8 
	ret int %C
}

uint %test9(uint %A) {
	%B = mul uint %A, 124
	%C = rem uint %B, 62 
	ret uint %C
}

int %test10(ubyte %c) {
        %tmp.1 = cast ubyte %c to int
        %tmp.2 = mul int %tmp.1, 3
        %tmp.3 = cast int %tmp.2 to ulong
        %tmp.5 = rem ulong %tmp.3, 3
        %tmp.6 = cast ulong %tmp.5 to int
        ret int %tmp.6
}

int %test11(int %i) {
        %tmp.1 = and int %i, -2
        %tmp.3 = mul int %tmp.1, 3
        %tmp.5 = rem int %tmp.3, 6
        ret int %tmp.5
}

