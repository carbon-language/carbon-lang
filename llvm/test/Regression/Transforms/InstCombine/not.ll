; This test makes sure that these instructions are properly eliminated.
;

; RUN: if as < %s | opt -instcombine -die | dis | grep xor
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

implementation

int "test1"(int %A) {
	%B = xor int %A
	%C = xor int %B
	ret int %C
}

