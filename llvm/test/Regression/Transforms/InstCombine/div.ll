; This test makes sure that div instructions are properly eliminated.
;

; RUN: if as < %s | opt -instcombine | dis | grep div
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

implementation

int %test1(int %A) {
	%B = div int %A, 1
	ret int %B
}
