; This test makes sure that these instructions are properly eliminated.
;

; RUN: if as < %s | opt -instcombine -dce | dis | grep sub
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

implementation

int "test1"(int %A) {
	%B = sub int %A, %A    ; ISA constant 0
	ret int %B
}

int "test2"(int %A) {
	%B = sub int %A, 0
	ret int %B
}

int "test3"(int %A) {
	%B = sub int 0, %A       ; B = -A
	%C = sub int 0, %B       ; C = -B = A
	ret int %C
}

int "test4"(int %A, int %x) {
	%B = sub int 0, %A
	%C = sub int %x, %B
	ret int %C
}

int "test5"(int %A, int %B, int %C) {
	%D = sub int %B, %C
	%E = sub int %A, %D
	ret int %E
}
