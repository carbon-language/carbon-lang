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


