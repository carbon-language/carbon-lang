; This test makes sure that these instructions are properly eliminated.
;

; RUN: if as < %s | opt -instcombine -die | dis | grep sub | grep -v 'sub int %Cok, %Bok'
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

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

