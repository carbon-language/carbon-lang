; This test makes sure that div instructions are properly eliminated.
;

; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep div

implementation

int %test1(int %A) {
	%B = div int %A, 1
	ret int %B
}

uint %test2(uint %A) {
	%B = div uint %A, 8   ; => Shift
	ret uint %B
}

int %test3(int %A) {
	%B = div int 0, %A    ; => 0, don't need to keep traps
	ret int %B
}

int %test4(int %A) {
	%B = div int %A, -1    ; 0-A
	ret int %B
}

uint %test5(uint %A) {
	%B = div uint %A, 4294967280
	%C = div uint %B, 4294967292
	ret uint %C
}

bool %test6(uint %A) {
	%B = div uint %A, 123
	%C = seteq uint %B, 0   ; A < 123
	ret bool %C
} 

bool %test7(uint %A) {
	%B = div uint %A, 10
	%C = seteq uint %B, 2    ; A >= 20 && A < 30
	ret bool %C
}

bool %test8(ubyte %A) {
	%B = div ubyte %A, 123
	%C = seteq ubyte %B, 2   ; A >= 246
	ret bool %C
} 

bool %test9(ubyte %A) {
	%B = div ubyte %A, 123
	%C = setne ubyte %B, 2   ; A < 246
	ret bool %C
} 

uint %test10(uint %X, bool %C) {
        %V = select bool %C, uint 64, uint 8
        %R = div uint %X, %V
        ret uint %R
}

