; This test makes sure that these instructions are properly eliminated.
;

; RUN: if as < %s | opt -instcombine | dis | grep rem
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

implementation

int "test1"(int %A) {
	%B = rem int %A, 1    ; ISA constant 0
	ret int %B
}

