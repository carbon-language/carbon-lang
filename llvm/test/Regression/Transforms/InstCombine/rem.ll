; This test makes sure that these instructions are properly eliminated.
;

; RUN: as < %s | opt -instcombine | dis | not grep rem

implementation

int %test1(int %A) {
	%B = rem int %A, 1    ; ISA constant 0
	ret int %B
}

int %test2(int %A) {          ; 0 % X = 0, we don't need ot preserve traps
	%B = rem int 0, %A
	ret int %B
}

uint %test3(uint %A) {
	%B = rem uint %A, 8   ; & 7
	ret uint %B
}
