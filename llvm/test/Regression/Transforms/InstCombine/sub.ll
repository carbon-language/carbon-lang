; This test makes sure that these instructions are properly eliminated.
;

; RUN: llvm-as < %s | opt -instcombine -die | llvm-dis | grep sub | not grep -v 'sub int %Cok, %Bok'

implementation

int %test1(int %A) {
	%B = sub int %A, %A    ; ISA constant 0
	ret int %B
}

int %test2(int %A) {
	%B = sub int %A, 0
	ret int %B
}

int %test3(int %A) {
	%B = sub int 0, %A       ; B = -A
	%C = sub int 0, %B       ; C = -B = A
	ret int %C
}

int %test4(int %A, int %x) {
	%B = sub int 0, %A
	%C = sub int %x, %B
	ret int %C
}

int %test5(int %A, int %Bok, int %Cok) {
	%D = sub int %Bok, %Cok
	%E = sub int %A, %D
	ret int %E
}

int %test6(int %A, int %B) {
	%C = and int %A, %B   ; A - (A & B) => A & ~B
	%D = sub int %A, %C
	ret int %D
}

int %test7(int %A) {
	%B = sub int -1, %A   ; B = ~A
	ret int %B
}

int %test8(int %A) {
        %B = mul int 9, %A
        %C = sub int %B, %A      ; C = 9*A-A == A*8 == A << 3
        ret int %C
}

int %test9(int %A) {
        %B = mul int 3, %A
        %C = sub int %A, %B      ; C = A-3*A == A*-2
        ret int %C
}

int %test10(int %A, int %B) {    ; -A*-B == A*B
	%C = sub int 0, %A
	%D = sub int 0, %B
	%E = mul int %C, %D
	ret int %E
}

int %test10(int %A) {    ; -A *c1 == A * -c1
	%C = sub int 0, %A
	%E = mul int %C, 7
	ret int %E
}

bool %test11(ubyte %A, ubyte %B) {
        %C = sub ubyte %A, %B
        %cD = setne ubyte %C, 0    ; == setne A, B
        ret bool %cD
}
