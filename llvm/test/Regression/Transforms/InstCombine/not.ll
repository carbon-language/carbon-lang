; This test makes sure that these instructions are properly eliminated.
;

; RUN: if as < %s | opt -instcombine -die | dis | grep xor
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

implementation

int %test1(int %A) {
	%B = xor int %A, -1
	%C = xor int %B, -1
	ret int %C
}

bool %test2(int %A, int %B) {
	%cond = setle int %A, %B     ; Can change into setge
	%Ret = xor bool %cond, true
	ret bool %Ret
}
